import os

import deepwave
import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# from deepwave import scalar
# from scipy.ndimage import gaussian_filter
from scipy.signal import butter
from dataclasses import dataclass
from collections import OrderedDict
from masthay_helpers.global_helpers import get_print, subdict

# from misfit_toys.data.dataset import get_data3

# from torch.nn import (
#     BCEWithLogitsLoss,
#     HuberLoss,
#     L1Loss,
#     SmoothL1Loss,
#     SoftMarginLoss,
# )
from torch.nn.parallel import DistributedDataParallel as DDP
from torchaudio.functional import biquad

# from misfit_toys.fwi.custom_losses import LeastSquares, CDFLoss
from misfit_toys.tccs.modules.seismic_data import (
    SeismicProp,
    Param,
    ParamConstrained,
    path_builder,
    chunk_and_deploy,
)

# from misfit_toys.data.dataset import towed_src, fixed_rec


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def get_file(name, *, rank="", path="out/parallel", ext=".pt"):
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


# Generate a velocity model constrained to be within a desired range
class Model(torch.nn.Module):
    def __init__(self, initial, min_vel, max_vel):
        super().__init__()
        self.min_vel = min_vel
        self.max_vel = max_vel
        self.model = torch.nn.Parameter(
            torch.logit((initial - min_vel) / (max_vel - min_vel))
        )

    def forward(self):
        return (
            torch.sigmoid(self.model) * (self.max_vel - self.min_vel)
            + self.min_vel
        )


def taper(x):
    # Taper the ends of traces
    return deepwave.common.cosine_taper_end(x, 100)


def filt(x, sos):
    return biquad(biquad(biquad(x, *sos[0]), *sos[1]), *sos[2])


def training_stages():
    def freq_preprocess(training, freq):
        sos = butter(6, freq, fs=1 / training.prop.module.meta.dt, output="sos")
        sos = [torch.tensor(sosi).to(training.obs_data.dtype) for sosi in sos]

        training.sos = sos

        training.obs_data_filt = filt(training.obs_data, sos)

        # training.report.obs_data_filt_record.append(
        #     training.custom.obs_data_filt
        # )
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
                "freqs",
                {
                    "data": [10, 15, 20, 25, 30],
                    "preprocess": freq_preprocess,
                    "postprocess": freq_postprocess,
                },
            ),
            (
                "epochs",
                {
                    "data": [0, 1],
                    "preprocess": epoch_preprocess,
                    "postprocess": epoch_postprocess,
                },
            ),
        ]
    )


@dataclass
class Training:
    rank: int
    world_size: int
    prop: torch.nn.Module
    obs_data: torch.Tensor
    loss_fn: torch.nn.Module
    optimizer: list
    training_stages: OrderedDict
    verbose: int = 1

    def __post_init__(self):
        self.optimizer_kwargs = self.optimizer
        self.optimizer = self.optimizer[0](
            self.prop.parameters(), **self.optimizer[1]
        )
        self.loss_record = []
        self.v_record = []
        self.out_record = []
        self.out_filt_record = []
        self.print, _ = get_print(_verbose=self.verbose)

    def _pre_train(self):
        pass

    def _post_train(self):
        save(torch.tensor(self.loss_record), "loss_record.pt", rank=self.rank)
        save(torch.stack(self.v_record), "vp_record.pt", rank=self.rank)
        save(torch.stack(self.out_record), "out_record.pt", rank=self.rank)
        save(
            torch.stack(self.out_filt_record),
            "out_filt_record.pt",
            rank=self.rank,
        )
        torch.distributed.barrier()
        # Plot
        if self.rank == 0:
            self.loss_record = torch.mean(
                torch.stack(
                    [
                        load("loss_record.pt", rank=rank)
                        for rank in range(self.world_size)
                    ]
                ),
                dim=0,
            )
            self.v_record = load("vp_record.pt", rank=0)
            out_record = torch.cat(
                [
                    load("out_record.pt", rank=rank)
                    for rank in range(self.world_size)
                ],
                dim=1,
            )
            self.out_filt_record = torch.cat(
                [
                    load("out_filt_record.pt", rank=rank)
                    for rank in range(self.world_size)
                ],
                dim=1,
            )

            save(self.loss_record, "loss_record.pt", rank="")
            save(self.v_record, "vp_record.pt", rank="")
            save(out_record, "out_record.pt", rank="")
            save(self.out_filt_record, "out_filt_record.pt", rank="")
        torch.distributed.barrier()
        cleanup()

    def train(self):
        self._pre_train()
        self._train()
        self._post_train()

    def _train(self):
        self.__recursive_train(
            level_data=self.training_stages,
            depth=0,
            max_depth=len(self.training_stages),
        )

    def __recursive_train(self, *, level_data, depth=0, max_depth=0):
        if depth == max_depth:
            self.step()  # Main training logic
            return

        level_name, level_info = list(level_data.items())[depth]
        data, preprocess, postprocess = (
            level_info["data"],
            level_info["preprocess"],
            level_info["postprocess"],
        )

        idt = "    " * depth
        for item in data:
            self.print(f"{idt}Preprocessing {level_name} {item}", verbose=2)
            preprocess(self, item)  # Assuming preprocess takes 'self'

            self.__recursive_train(
                level_data=level_data, depth=depth + 1, max_depth=max_depth
            )

            self.print(f"{idt}Postprocessing {level_name} {item}", verbose=2)
            postprocess(self, item)

    def _step(self):
        self.out = self.prop(1)
        self.out_filt = filt(taper(self.out[-1]), self.sos)
        self.loss = 1e6 * self.loss_fn(self.out_filt, self.obs_data_filt)
        self.loss.backward()
        return self.loss

    def step(self):
        num_calls = 0

        def closure():
            nonlocal num_calls
            num_calls += 1
            self.optimizer.zero_grad()
            self._step()
            if num_calls == 1:
                self.update_records()
            return self.loss

        self.optimizer.step(closure)

    def preprocess_freqs(self, *, cutoff_freq):
        self.sos = butter(6, cutoff_freq, fs=1 / self.dt, output="sos")
        self.sos = [
            torch.tensor(sosi).to(self.obs_data.dtype).to(self.rank)
            for sosi in self.sos
        ]

        self.obs_data_filt = filt(self.obs_data, self.sos)

    def get_epoch(self, i, j):
        return j + i * self.n_epochs

    def update_records(self):
        self.loss_record.append(self.loss)
        self.v_record.append(self.prop.module.vp().detach().cpu())
        self.out_record.append(self.out[-1].detach().cpu())
        self.out_filt_record.append(self.out_filt.detach().cpu())
        # print(
        #     f"Epoch={self.get_epoch(self.freq_no, self.epoch)},"
        #     f" Loss={self.loss.item()}, rank={self.rank}",
        #     flush=True,
        # )

    def reset_optimizer(self):
        self.optimizer = self.optimizer_kwargs[0](
            self.prop.parameters(), **self.optimizer_kwargs[1]
        )


def run_rank(rank, world_size):
    print(f"Running DDP on rank {rank} / {world_size}.")
    setup(rank, world_size)

    data = path_builder(
        "conda/data/marmousi/deepwave_example/shots16",
        remap={"vp_init": "vp"},
        vp_init=ParamConstrained.delay_init(
            minv=1000, maxv=2500, requires_grad=True
        ),
        src_amp_y=Param.delay_init(requires_grad=False),
        obs_data=None,
        src_loc_y=None,
        rec_loc_y=None,
    )
    data["obs_data"] = taper(data["obs_data"])
    data = chunk_and_deploy(
        rank,
        world_size,
        data=data,
        chunk_keys={
            "tensors": ["obs_data", "src_loc_y", "rec_loc_y"],
            "params": ["src_amp_y"],
        },
    )

    prop_data = subdict(data, exclude=["obs_data"])
    prop = SeismicProp(
        **prop_data, max_vel=2500, pml_freq=data["meta"].freq, time_pad_frac=0.2
    ).to(rank)
    prop = DDP(prop, device_ids=[rank])

    loss_fn = torch.nn.MSELoss()

    train = Training(
        rank=rank,
        world_size=world_size,
        prop=prop,
        obs_data=data["obs_data"],
        loss_fn=loss_fn,
        optimizer=[torch.optim.LBFGS, {}],
        verbose=2,
        training_stages=training_stages(),
    )
    train.train()


def run(world_size):
    mp.spawn(run_rank, args=(world_size,), nprocs=world_size, join=True)


if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    run(n_gpus)
