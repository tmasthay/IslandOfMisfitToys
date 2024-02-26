import logging
import os
import sys
from collections import OrderedDict
from time import time

import hydra
import matplotlib.pyplot as plt
import torch
import torch.multiprocessing as mp
from helpers import StdoutLogger, W2Loss, setup_logger
from mh.core import DotDict, convert_dictconfig, exec_imports, hydra_out
from mh.core_legacy import subdict
from mh.typlotlib import apply_subplot, get_frames_bool, save_frames
from omegaconf import DictConfig
from scipy.signal import butter
from torch.nn.parallel import DistributedDataParallel as DDP

from misfit_toys.fwi.loss.tikhonov import TikhonovLoss, lin_reg_tmp
from misfit_toys.fwi.seismic_data import (
    Param,
    ParamConstrained,
    SeismicProp,
    path_builder,
)
from misfit_toys.fwi.training import Training
from misfit_toys.utils import (
    bool_slice,
    chunk_and_deploy,
    clean_idx,
    filt,
    setup,
    taper,
    apply,
    d2cpu,
)

torch.set_printoptions(precision=3, sci_mode=True, threshold=5, linewidth=10)


# Main function for training on each rank
def run_rank(rank: int, world_size: int, c: DotDict) -> None:
    print(f"Running DDP on rank {rank} / {world_size}.")
    setup(rank, world_size)

    if c.get('dupe', False):
        out_file = f'{c.rank_out}_{rank}.out'
        err_file = f'{c.rank_out}_{rank}.err'

        print(
            f'Now duping stdout and stderr on rank {rank} to files below:\n\n'
            f'    {out_file}\n\n    {err_file}.',
            flush=True,
        )
        sys.stdout = open(out_file, 'w')
        sys.stderr = open(err_file, 'w')

    start_pre = time()
    c = resolve(c, relax=True)
    print(f"Preprocessing took {time() - start_pre:.2f} seconds.", flush=True)
    # Build data for marmousi model
    # raise ValueError(c.data.path)
    data = path_builder(
        c.data.path,
        remap={"vp_init": "vp"},
        vp_init=ParamConstrained.delay_init(
            minv=c.preprocess.minv, maxv=c.preprocess.maxv, requires_grad=True
        ),
        src_amp_y=Param.delay_init(requires_grad=False),
        obs_data=None,
        src_loc_y=None,
        rec_loc_y=None,
    )

    # preprocess data like Alan and then deploy slices onto GPUs
    # data["obs_data"] = taper(data["obs_data"])
    data = chunk_and_deploy(
        rank,
        world_size,
        data=data,
        chunk_keys={
            "tensors": ["obs_data", "src_loc_y", "rec_loc_y"],
            "params": ["src_amp_y"],
        },
    )

    if torch.isnan(data['vp'].p).any():
        raise ValueError("NaNs in vp")

    # Build seismic propagation module and wrap in DDP
    prop_data = subdict(data, exc=["obs_data"])
    c.obs_data = data["obs_data"]
    c.prop = SeismicProp(
        **prop_data,
        max_vel=c.preprocess.maxv,
        pml_freq=data["meta"].freq,
        time_pad_frac=0.2,
    ).to(rank)
    c.prop = DDP(c.prop, device_ids=[rank])
    c = resolve(c, relax=False)
    # loss_fn = c.train.loss.type(
    #     weights=c.prop.module.vp,
    #     alpha=get_reg_strength(c),
    #     max_iters=c.train.max_iters,
    # )
    loss_fn = apply(c.train.loss, c)
    optimizer = apply(c.train.optimizer, c)
    step = apply(c.train.step, c)
    training_stages = apply(c.train.stages, c)
    pre_time = time() - start_pre
    print(f"Preprocess time rank {rank}: {pre_time:.2f} seconds.", flush=True)
    # loss_fn = c.train.loss.tik.type(
    #     weights=c.prop.module.vp,
    #     alpha=lin_reg_tmp(c),
    #     max_iters=c.train.max_iters,
    # )
    # Define the training object
    train = Training(
        rank=rank,
        world_size=world_size,
        prop=c.prop,
        obs_data=data["obs_data"],
        loss_fn=loss_fn,
        optimizer=optimizer,
        verbose=1,
        report_spec={
            'path': os.path.join(os.getcwd(), 'out'),
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
            # 'out_filt': {
            #     'update': lambda x: d2cpu(x.out_filt),
            #     'reduce': lambda x: torch.cat(x, dim=1),
            #     'presave': torch.stack,
            # },
        },
        _step=step,
        _build_training_stages=training_stages,
    )
    train_start = time()
    train.train()
    train_time = time() - train_start
    print(f"Train time rank {rank}: {train_time:.2f} seconds.", flush=True)


# Main function for spawning ranks
def run(world_size: int, c: DotDict) -> None:
    mp.spawn(run_rank, args=(world_size, c), nprocs=world_size, join=True)


def preprocess_cfg(cfg: DictConfig) -> DotDict:
    c = convert_dictconfig(cfg)
    for k, v in c.plt.items():
        c[f'plt.{k}.save.path'] = hydra_out(v.save.path)
    c.data.path = c.data.path.replace('conda', os.environ['CONDA_PREFIX'])
    c.rank_out = hydra_out(c.get('rank_out', 'rank'))
    return c


def resolve(c: DotDict, relax) -> DotDict:
    c = exec_imports(c)
    c.self_ref_resolve(gbl=globals(), lcl=locals(), relax=relax)
    return c


def plotter(*, data, idx, fig, axes, c):
    # plt.imshow(data[idx], **c.plt.vp)
    plt.clf()
    apply_subplot(
        data=data[idx],
        cfg=c.plt.vp,
        name='vp',
        layer='main',
        title=("", f"\n{clean_idx(idx)}"),
    )
    apply_subplot(
        data=c.rel_diff[idx], cfg=c.plt.vp, name='rel_diff', layer='main'
    )
    apply_subplot(
        data=c.vp_true.squeeze(), cfg=c.plt.vp, name='vp_true', layer='main'
    )
    plt.subplot(*c.plt.vp.sub.shape, 4)
    plt.plot(c.loss)
    plt.scatter(idx[0], c.loss[idx[0]], color='r', s=100, marker='o')
    plt.title('Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.tight_layout()
    return {'c': c}


@hydra.main(config_path="cfg", config_name="cfg", version_base=None)
def main(cfg: DictConfig) -> None:
    c = preprocess_cfg(cfg)

    out_dir = os.path.join(os.path.dirname(__file__), 'out')

    def get_data():
        files = [
            os.path.join(out_dir, e)
            for e in os.listdir(out_dir)
            if e.endswith('_record.pt')
        ]
        keys = [e.replace('_record.pt', '').split('/')[-1] for e in files]
        return DotDict({k: torch.load(f) for k, f in zip(keys, files)})

    data = get_data()
    if not data or c.train.retrain:
        n_gpus = torch.cuda.device_count()
        run(n_gpus, c)
        data = get_data()

    c.plt = resolve(c.plt, relax=False)
    iter = bool_slice(*data.vp.shape, **c.plt.vp.iter)
    fig, axes = plt.subplots(*c.plt.vp.sub.shape, **c.plt.vp.sub.kw)
    if c.plt.vp.sub.adjust:
        plt.subplots_adjust(**c.plt.vp.sub.adjust)
    # data.vp = data.vp.permute(0, 2, 1)
    # input(data.keys())
    vp_true = torch.load(
        os.path.join(
            c.data.path,
            "vp_true.pt",
        )
    )
    # c.vp_true = vp_true.T.unsqueeze(0)
    c.vp_true = vp_true.unsqueeze(0)
    c.rel_diff = data.vp - c.vp_true
    c.rel_diff = c.rel_diff / torch.abs(c.vp_true) * 100.0
    c.loss = data.loss
    frames = get_frames_bool(
        data=data.vp,
        iter=iter,
        fig=fig,
        axes=axes,
        plotter=plotter,
        c=c,
    )
    save_frames(frames, **c.plt.vp.save)

    for k, v in c.plt.items():
        print(f"Plot {k} stored in {v.save.path}")

    print('To see all output run the following in terminal:\n')
    print(f'    cd {hydra_out("")}')


# Run the script from command line
if __name__ == "__main__":
    main()
