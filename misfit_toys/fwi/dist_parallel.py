import os
import torch
from torchaudio.functional import biquad
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from scipy.ndimage import gaussian_filter
from scipy.signal import butter
import matplotlib.pyplot as plt
import deepwave
from deepwave import scalar
from ..utils import *
from .modules.seismic_data import SeismicData
from .modules.models import Model, Prop
from .modules.visual import make_plots
from .modules.training import Training
from .modules.distribution import Distribution, setup, cleanup

def run_rank(rank, world_size):
    print(f"Running DDP on rank {rank} / {world_size}.")
    setup(rank, world_size)

    data = SeismicData()

    #source locations
    # src_loc = towed_src(
    #     n_shots=data.n_shots,
    #     src_per_shot=data.src_per_shot,
    #     d_src=20,
    #     fst_src=10,
    #     src_depth=2,
    #     d_intra_shot=0
    # )

    # #receiver locations
    # rec_loc = fixed_rec(
    #     n_shots=data.n_shots,
    #     rec_per_shot=data.rec_per_shot,
    #     d_rec=6,
    #     rec_depth=2,
    #     fst_rec=0
    # )

    # # source amplitudes
    # src_amp = (
    #     (dw.wavelets.ricker(data.freq, data.nt, data.dt, data.peak_time))
    #     .repeat(data.n_shots, data.src_per_shot, 1)
    # )

    model = Model(data.v_init, 1000, 2500)

    #Setup distribution onto multiple GPUs
    d = Distribution(rank=rank, world_size=world_size)
    prop, data.obs_data, data.src_amp_y, data.src_loc, data.rec_loc = \
        d.setup_distribution(
            obs_data=data.obs_data,
            src_amp=data.src_amp_y,
            src_loc=data.src_loc,
            rec_loc=data.rec_loc,
            model=model,
            dx=data.dx,
            dt=data.dt,
            freq=data.freq
        )

    #Perform training
    train_obj = Training(
        prop=prop,
        src_amp=data.src_amp_y,
        src_loc=data.src_loc,
        rec_loc=data.rec_loc,
        obs_data=data.obs_data,
        dt=data.dt,
        rank=rank
    )

    # Plot
    if rank == 0:
        make_plots(v_true=data.v_true, v_init=data.v_init, model=model)

    cleanup()


def run(world_size):
    mp.spawn(run_rank,
             args=(world_size,),
             nprocs=world_size,
             join=True)

def main():
    n_gpus = torch.cuda.device_count()
    run(n_gpus)

if __name__ == "__main__":
    main()