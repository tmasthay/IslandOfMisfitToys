import os

import deepwave
import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from deepwave import scalar
from scipy.ndimage import gaussian_filter
from scipy.signal import butter
from torch.nn.parallel import DistributedDataParallel as DDP
from torchaudio.functional import biquad

from misfit_toys.fwi.custom_losses import LeastSquares, CDFLoss
from misfit_toys.tccs.modules.seismic_data import (
    SeismicProp,
    Param,
    ParamConstrained,
)
from misfit_toys.data.dataset import towed_src, fixed_rec
from collections import OrderedDict
from misfit_toys.tccs.modules.training import Training
from tabulate import tabulate as tab
from rich.live import Live
from rich.table import Table


def setup(rank, world_size):
    """_summary_

    Args:
        rank (_type_): _description_
        world_size (_type_): _description_
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    """_summary_
    Args:
        oyoyoyoy
    """
    dist.destroy_process_group()


def get_file(name, *, rank="", path="out/parallel", ext=".pt"):
    """_summary_

    Args:
        name (_type_): _description_
        rank (str, optional): _description_. Defaults to "".
        path (str, optional): _description_. Defaults to "out/parallel".
        ext (str, optional): _description_. Defaults to ".pt".

    Returns:
        _type_: _description_
    """
    ext = "." + ext.replace(".", "")
    name = name.replace(ext, "")
    if rank != "":
        rank = f"_{rank}"
    return os.path.join(os.path.dirname(__file__), path, f"{name}{rank}{ext}")


def load(name, *, rank="", path="out/parallel", ext=".pt"):
    return torch.load(get_file(name, rank=rank, path=path, ext=".pt"))


def save(tensor, name, *, rank="", path="out/parallel", ext=".pt"):
    torch.save(tensor, get_file(name, rank=rank, path=path, ext=".pt"))


def savefig(name, *, path="out/parallel", ext=".pt"):
    plt.savefig(get_file(name, rank="", path=path, ext=ext))


def filt(x, sos):
    return biquad(biquad(biquad(x, *sos[0]), *sos[1]), *sos[2])


def taper(x):
    # Taper the ends of traces
    return deepwave.common.cosine_taper_end(x, 100)


def pre_train(training):
    training.report.set('obs_data_filt_record', [])
    training.report.set('obs_data_record', [])


def training_stages():
    def freq_preprocess(training, freq):
        sos = butter(
            6, freq, fs=1 / training.dist_prop.module.meta.dt, output="sos"
        )
        sos = [
            torch.tensor(sosi).to(training.dist_prop.module.obs_data.dtype)
            for sosi in sos
        ]

        training.custom.sos = sos

        training.custom.obs_data_filt = filt(
            training.dist_prop.module.obs_data, sos
        )

        training.report.obs_data_filt_record.append(
            training.custom.obs_data_filt
        )
        training.reset_optimizer()

    def freq_postprocess(training, freq):
        pass

    def epoch_preprocess(training, epoch):
        pass

    def epoch_postprocess(training, epoch):
        pass

    return OrderedDict(
        [
            (
                'freqs',
                {
                    'data': [10, 15, 20, 25, 30],
                    'preprocess': freq_preprocess,
                    'postprocess': freq_postprocess,
                },
            ),
            (
                'epochs',
                {
                    'data': [0, 1],
                    'preprocess': epoch_preprocess,
                    'postprocess': epoch_postprocess,
                },
            ),
        ]
    )


def step(training):
    out = training.dist_prop(1)
    out_filt = filt(out[-1], training.custom.sos)
    loss = 1e6 * training.loss(out_filt, training.custom.obs_data_filt)
    other_info = {
        'out_record': out[-1],
        'out_filt_record': out_filt,
        'vp_record': training.dist_prop.module.vp(),
    }
    return loss, other_info


# # Generate a velocity model constrained to be within a desired range
# class Model(torch.nn.Module):
#     def __init__(self, initial, min_vel, max_vel):
#         super().__init__()
#         self.min_vel = min_vel
#         self.max_vel = max_vel
#         self.model = torch.nn.Parameter(
#             torch.logit((initial - min_vel) / (max_vel - min_vel))
#         )

#     def forward(self):
#         return (
#             torch.sigmoid(self.model) * (self.max_vel - self.min_vel)
#             + self.min_vel
#         )


# class Prop(torch.nn.Module):
#     def __init__(self, model, dx, dt, freq):
#         super().__init__()
#         self.model = model
#         self.dx = dx
#         self.dt = dt
#         self.freq = freq

#     def forward(self, source_amplitudes, source_locations, receiver_locations):
#         v = self.model()
#         return scalar(
#             v,
#             self.dx,
#             self.dt,
#             source_amplitudes=source_amplitudes,
#             source_locations=source_locations,
#             receiver_locations=receiver_locations,
#             max_vel=2500,
#             pml_freq=self.freq,
#             time_pad_frac=0.2,
#         )


def run_rank(rank, world_size):
    print(f"Running DDP on rank {rank} / {world_size}.")
    setup(rank, world_size)
    ny = 2301
    nx = 751
    dx = 4.0
    v_true = load("vp.pt", path="out/base")

    # Select portion of model for inversion
    ny = 600
    nx = 250
    v_true = v_true[:ny, :nx]

    # Smooth to use as starting model
    v_init = torch.tensor(1.0 / gaussian_filter(1.0 / v_true.numpy(), 40))

    n_shots = 115

    n_sources_per_shot = 1
    d_source = 20  # 20 * 4m = 80m
    first_source = 10  # 10 * 4m = 40m
    source_depth = 2  # 2 * 4m = 8m

    n_receivers_per_shot = 384
    d_receiver = 6  # 6 * 4m = 24m
    first_receiver = 0  # 0 * 4m = 0m
    receiver_depth = 2  # 2 * 4m = 8m

    freq = 25
    nt = 750
    dt = 0.004
    peak_time = 1.5 / freq

    observed_data = load("obs_data.pt", path="out/base")

    # Select portion of data for inversion
    n_shots = 16
    n_receivers_per_shot = 100
    nt = 300
    observed_data = taper(observed_data[:n_shots, :n_receivers_per_shot, :nt])

    alan_src = torch.zeros(n_shots, n_sources_per_shot, 2, dtype=torch.long)
    alan_src[..., 1] = source_depth
    alan_src[:, 0, 0] = torch.arange(n_shots) * d_source + first_source
    source_locations = towed_src(
        n_shots=n_shots,
        src_per_shot=n_sources_per_shot,
        src_depth=source_depth,
        d_src=d_source,
        fst_src=first_source,
        d_intra_shot=0,
    )
    if torch.max(source_locations - alan_src) > 0:
        raise ValueError(
            "towed_src and source_locations do not match,"
            f" norm={torch.max(source_locations - alan_src)}"
        )

    # receiver_locations
    rec_alan = torch.zeros(n_shots, n_receivers_per_shot, 2, dtype=torch.long)
    rec_alan[..., 1] = receiver_depth
    rec_alan[:, :, 0] = (
        torch.arange(n_receivers_per_shot) * d_receiver + first_receiver
    ).repeat(n_shots, 1)
    receiver_locations = fixed_rec(
        n_shots=n_shots,
        rec_per_shot=n_receivers_per_shot,
        rec_depth=receiver_depth,
        d_rec=d_receiver,
        fst_rec=first_receiver,
    )
    if torch.max(receiver_locations - rec_alan) > 0:
        raise ValueError(
            "fixed_rec and receiver_locations do not match,"
            f" norm={torch.max(receiver_locations - rec_alan)}"
        )

    # source_amplitudes
    source_amplitudes = (
        deepwave.wavelets.ricker(freq, nt, dt, peak_time)
    ).repeat(n_shots, n_sources_per_shot, 1)

    source_amplitudes = Param(p=source_amplitudes, requires_grad=False)
    # source_locations = Param(source_locations, requires_grad=False)
    # receiver_locations = Param(receiver_locations, requires_grad=False)

    observed_data = torch.chunk(observed_data, world_size)[rank].to(rank)
    source_amplitudes.p.data = torch.chunk(
        source_amplitudes.p.data, world_size
    )[rank].to(rank)
    source_locations = torch.chunk(source_locations, world_size)[rank].to(rank)
    receiver_locations = torch.chunk(receiver_locations, world_size)[rank].to(
        rank
    )

    # model = Model(v_init, 1000, 2500)
    vp = ParamConstrained(
        p=v_init,
        minv=1000,
        maxv=2500,
        requires_grad=True,
    )
    prop = SeismicProp(
        vp=vp,
        model='acoustic',
        dx=dx,
        dt=dt,
        obs_data=observed_data,
        src_amp_y=source_amplitudes,
        src_loc_y=source_locations,
        rec_loc_y=receiver_locations,
        max_vel=2500,
        pml_freq=freq,
        time_pad_frac=0.2,
        meta={'dt': dt, 'dx': dx, 'nt': nt, 'nx': nx, 'ny': ny},
    ).to(rank)
    prop = DDP(prop, device_ids=[rank])

    training = Training(
        step=step,
        pre_train=pre_train,
        post_train=None,
        dist_prop=prop,
        rank=rank,
        world_size=world_size,
        verbose=1,
        optimizer=(torch.optim.LBFGS, {}),
        scheduler=None,
        loss=torch.nn.MSELoss(),
        training_stages=training_stages(),
    )

    # Setup optimiser to perform inversion
    # loss_fn = torch.nn.MSELoss()
    # # loss_fn = LeastSquares()
    # # loss_fn = HuberLoss()
    # # loss_fn = L1Loss()
    # # loss_fn = BCEWithLogitsLoss()
    # # loss_fn = SoftMarginLoss()
    # # def renorm_func(x):
    # #     return x**2

    # # loss_fn = CDFLoss(renorm=renorm_func)

    # # Run optimisation/inversion
    # n_epochs = 2

    # loss_record = []
    # v_record = []
    # out_record = []
    # out_filt_record = []

    # freqs = [10, 15, 20, 25, 30]
    # n_freqs = len(freqs)
    # get_epoch = lambda i, j: i * n_epochs + j
    # for i, cutoff_freq in enumerate(freqs):
    #     sos = butter(6, cutoff_freq, fs=1 / dt, output="sos")
    #     sos = [
    #         torch.tensor(sosi).to(observed_data.dtype).to(rank) for sosi in sos
    #     ]

    #     def filt(x):
    #         return biquad(biquad(biquad(x, *sos[0]), *sos[1]), *sos[2])

    #     observed_data_filt = filt(observed_data)
    #     optimiser = torch.optim.LBFGS(prop.parameters())
    #     for epoch in range(n_epochs):
    #         num_calls = 0

    #         def closure():
    #             nonlocal num_calls
    #             num_calls += 1
    #             optimiser.zero_grad()
    #             out = prop(1)
    #             out_filt = filt(taper(out[-1]))
    #             loss = 1e6 * loss_fn(out_filt, observed_data_filt)
    #             loss.backward()
    #             if num_calls == 1:
    #                 loss_record.append(loss)
    #                 v_record.append(prop.module.vp().detach().cpu())
    #                 out_record.append(out[-1].detach().cpu())
    #                 out_filt_record.append(out_filt.detach().cpu())
    #                 print(
    #                     f"Epoch={get_epoch(i, epoch)}, Loss={loss.item()},"
    #                     f" rank={rank}",
    #                     flush=True,
    #                 )
    #             return loss

    #         optimiser.step(closure)

    training.train()
    save(torch.tensor(training.report.loss_record), "loss_record.pt", rank=rank)

    # raise ValueError(f'length={len(training.report.vp_record)}')
    save(torch.stack(training.report.vp_record), "vp_record.pt", rank=rank)
    save(torch.stack(training.report.out_record), "out_record.pt", rank=rank)
    save(
        torch.stack(training.report.out_filt_record),
        "out_filt_record.pt",
        rank=rank,
    )

    torch.distributed.barrier()
    # Plot
    if rank == 0:
        loss_record = torch.mean(
            torch.stack(
                [
                    load("loss_record.pt", rank=rank)
                    for rank in range(world_size)
                ]
            )
        )
        v_record = load("vp_record.pt", rank=0)
        out_record = torch.cat(
            [load("out_record.pt", rank=rank) for rank in range(world_size)]
        )
        out_filt_record = torch.cat(
            [
                load("out_filt_record.pt", rank=rank)
                for rank in range(world_size)
            ]
        )

        save(loss_record, "loss_record.pt", rank="")
        save(v_record, "vp_record.pt", rank="")
        save(out_record, "out_record.pt", rank="")
        save(out_filt_record, "out_filt_record.pt", rank="")

        # v = prop.module.vp()
        # vmin = v_true.min()
        # vmax = v_true.max()
        # _, ax = plt.subplots(3, figsize=(10.5, 10.5), sharex=True, sharey=True)
        # ax[0].imshow(
        #     v_init.cpu().T, aspect="auto", cmap="gray", vmin=vmin, vmax=vmax
        # )
        # ax[0].set_title("Initial")
        # ax[1].imshow(
        #     v.detach().cpu().T, aspect="auto", cmap="gray", vmin=vmin, vmax=vmax
        # )
        # ax[1].set_title("Out")
        # ax[2].imshow(
        #     v_true.cpu().T, aspect="auto", cmap="gray", vmin=vmin, vmax=vmax
        # )
        # ax[2].set_title("True")
        # plt.tight_layout()
        # savefig("example_distributed_ddp", ext="jpg")

        # v.detach().cpu().numpy().tofile("marmousi_v_inv.bin")
    cleanup()


def run(world_size):
    mp.spawn(run_rank, args=(world_size,), nprocs=world_size, join=True)


if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    run(n_gpus)
