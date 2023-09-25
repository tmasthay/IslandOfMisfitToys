from ..utils import *
from .modules.seismic_data import SeismicProp
from .modules.visual import make_plots
from .modules.training import Training
from .modules.distribution import Distribution, setup, cleanup
from .modules.models import Param, ParamConstrained

import torch
import torch.multiprocessing as mp
import copy
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'


def run_rank(rank, world_size):
    print(f"Running DDP on rank {rank} / {world_size}.")
    setup(rank, world_size)

    # fetch data from deepwave_example
    #  NOTE: vp_init is very slightly different from Alan's code for some
    #    reason. I don't know why, but this will lead to slightly different
    #    final inversion results, and significantly different loss
    #    evaluations. However, those loss evaluations are still within the
    #    same order of magnitude, e.g. final value is roughly 8.0 for this
    #    script as of Sep 20, 2023, and roughly 2.0 for Alan's code.
    # NOTE FOLLOWUP:
    #   I might be wrong above, given that the closure gets evaluated a
    #     ton of times. Loss plots look like they may match.
    prop = SeismicProp(
        path='conda/data/marmousi/deepwave_example/shots16',
        vp_prmzt=ParamConstrained.delay_init(
            requires_grad=True, minv=1000, maxv=2500
        ),
        extra_forward_args={'time_pad_frac': 0.2},
    )
    prop.obs_data = taper(prop.obs_data, 100)
    torch.save(prop.obs_data, f'tyler_obs_data_{rank}.pt')
    vp_init = copy.deepcopy(prop.vp().detach().cpu())

    # Setup distribution onto multiple GPUs
    distribution = Distribution(rank=rank, world_size=world_size, prop=prop)

    # Perform training
    train_obj = Training(distribution=distribution)
    train_obj.train()

    # Plot
    if rank == 0:
        make_plots(v_true=prop.vp_true, v_init=vp_init, vp=prop.vp)

    cleanup()


def run(world_size):
    mp.spawn(run_rank, args=(world_size,), nprocs=world_size, join=True)


def main():
    n_gpus = torch.cuda.device_count()
    run(n_gpus)


if __name__ == "__main__":
    main()
