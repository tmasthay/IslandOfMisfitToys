import os

import deepwave
import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from scipy.signal import butter
from dataclasses import dataclass
from collections import OrderedDict
from masthay_helpers.global_helpers import (
    get_print,
    subdict,
    DotDict,
    flip_dict,
)
from torch.optim.lr_scheduler import ChainedScheduler
from misfit_toys.utils import setup, filt, taper
from misfit_toys.tccs.modules.training import Training

from torch.nn.parallel import DistributedDataParallel as DDP
from torchaudio.functional import biquad
from misfit_toys.tccs.modules.seismic_data import (
    SeismicProp,
    Param,
    ParamConstrained,
    path_builder,
    chunk_and_deploy,
)


# def setup(rank, world_size):
#     os.environ["MASTER_ADDR"] = "localhost"
#     os.environ["MASTER_PORT"] = "12355"

#     # initialize the process group
#     dist.init_process_group("nccl", rank=rank, world_size=world_size)
#     torch.cuda.set_device(rank)


# def cleanup():
#     dist.destroy_process_group()

# def get_file(name, *, rank="", path="out/parallel", ext=".pt"):
#     ext = "." + ext.replace(".", "")
#     name = name.replace(ext, "")
#     if rank != "":
#         rank = f"_{rank}"
#     return os.path.join(os.path.dirname(__file__), path, f"{name}{rank}{ext}")


# def load(name, *, rank="", path="out/parallel", ext=".pt"):
#     return torch.load(get_file(name, rank=rank, path=path, ext=".pt"))


# def load_all(name, *, world_size=0, path='out/parallel', ext='.pt'):
#     if world_size == -1:
#         return load(name, rank='', path=path, ext=ext)
#     else:
#         return [
#             load(name, rank=rank, path=path, ext=ext)
#             for rank in range(world_size)
#         ]


# def save(tensor, name, *, rank="", path="out/parallel", ext=".pt"):
#     torch.save(tensor, get_file(name, rank=rank, path=path, ext=".pt"))


# def savefig(name, *, path="out/parallel", ext=".pt"):
#     plt.savefig(get_file(name, rank="", path=path, ext=ext))


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


def _step(self):
    self.out = self.prop(1)
    self.out_filt = filt(taper(self.out[-1]), self.sos)
    self.loss = 1e6 * self.loss_fn(self.out_filt, self.obs_data_filt)
    self.loss.backward()
    return self.loss


def d2cpu(x):
    return x.detach().cpu()


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
        report_spec={
            'path': os.path.join(os.path.dirname(__file__), 'out', 'parallel'),
            'loss': {
                'update': lambda x: d2cpu(x.loss),
                'reduce': lambda x: torch.mean(torch.stack(x), dim=0),
                'presave': torch.tensor,
            },
            'vp': {
                'update': lambda x: d2cpu(x.prop.module.vp()),
                'reduce': lambda x: x[0],
                'presave': torch.stack,
            },
            'out': {
                'update': lambda x: d2cpu(x.out[-1]),
                'reduce': lambda x: torch.cat(x, dim=1),
                'presave': torch.stack,
            },
            'out_filt': {
                'update': lambda x: d2cpu(x.out_filt),
                'reduce': lambda x: torch.cat(x, dim=1),
                'presave': torch.stack,
            },
        },
        _step=_step,
        _build_training_stages=training_stages,
    )
    train.train()


def run(world_size):
    mp.spawn(run_rank, args=(world_size,), nprocs=world_size, join=True)


def main():
    n_gpus = torch.cuda.device_count()
    run(n_gpus)


if __name__ == "__main__":
    main()
