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
from mh.core import DotDict, convert_dictconfig, hydra_out
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
    resolve,
)
from misfit_toys.swiffer import dupe

torch.set_printoptions(precision=3, sci_mode=True, threshold=5, linewidth=10)


def check_keys(c, data):
    pre = c.data.preprocess
    required = pre.required_fields
    if 'remap' in pre.path_builder_kw.keys():
        for k, v in pre.path_builder_kw.remap.items():
            if k in required:
                required.remove(k)
                required.append(v)
    if not set(required).issubset(data.keys()):
        raise ValueError(
            f"Missing required fields in data\n "
            f"    Required: {pre.required_fields}\n"
            f"    Found: {list(data.keys())}"
        )


# Main function for training on each rank
def run_rank(rank: int, world_size: int, c: DotDict) -> None:
    print(f"Running DDP on rank {rank} / {world_size}.")
    setup(rank, world_size)

    if c.get('dupe', False):
        dupe(f'{c.rank_out}_{rank}')

    start_pre = time()
    c = resolve(c, relax=True)
    print(f"Preprocessing took {time() - start_pre:.2f} seconds.", flush=True)

    for k, v in c.data.preprocess.path_builder_kw.items():
        if isinstance(v, dict) or isinstance(v, DotDict):
            if 'type' in v.keys():
                c.data.preprocess.path_builder_kw[k] = apply(
                    c.data.preprocess.path_builder_kw[k], c
                )

    c['runtime.data'] = path_builder(
        c.data.path, **c.data.preprocess.path_builder_kw
    )

    check_keys(c, c.runtime.data)

    c.runtime.data = DotDict(
        chunk_and_deploy(
            rank,
            world_size,
            data=c.runtime.data,
            chunk_keys=c.data.preprocess.chunk_keys,
        )
    )

    if torch.isnan(c.runtime.data.vp.p).any():
        raise ValueError("NaNs in vp")

    # Build seismic propagation module and wrap in DDP
    prop_data = subdict(c.runtime.data, exc=["obs_data"])
    c.obs_data = c.runtime.data.obs_data
    c['runtime.prop'] = SeismicProp(
        **prop_data,
        max_vel=c.data.preprocess.maxv,
        pml_freq=c.runtime.data.meta.freq,
        time_pad_frac=c.data.preprocess.time_pad_frac,
    ).to(rank)
    c.runtime.prop = DDP(c.runtime.prop, device_ids=[rank])

    c = resolve(c, relax=False)
    loss_fn = apply(c.train.loss, c)
    optimizer = apply(c.train.optimizer, c)
    step = apply(c.train.step, c)
    training_stages = apply(c.train.stages, c)

    pre_time = time() - start_pre
    print(f"Preprocess time rank {rank}: {pre_time:.2f} seconds.", flush=True)

    train = Training(
        rank=rank,
        world_size=world_size,
        prop=c.runtime.prop,
        obs_data=c.runtime.data.obs_data,
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

    vp_true = torch.load(
        os.path.join(
            c.data.path,
            "vp_true.pt",
        )
    )
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


# u = {
#     'np': 'self.prop.module.meta.nt',
#     'dupe': True,
#     'plt': {
#         'vp': {
#             'sub': {
#                 'shape': [2, 2],
#                 'kw': {'figsize': [10, 10]},
#                 'adjust': {'hspace': 0.5, 'wspace': 0.5},
#             },
#             'iter': {'none_dims': [-2, -1]},
#             'save': {
#                 'path': '/home/tyler/Documents/repos/IslandOfMisfitToys/misfit_toys/examples/hydra/outputs/2024-02-26/22-38-56/figs/vp.gif',
#                 'movie_format': 'gif',
#                 'duration': 1000,
#             },
#             'order': ['vp', 'vp_true', 'rel_diff'],
#             'plts': {
#                 'vp': {
#                     'main': {
#                         'filt': "<function <lambda> at 0x7f3a4705c9d0>",
#                         'opts': {'cmap': 'gray', 'aspect': 'auto'},
#                         'title': '$v_p$',
#                         'type': 'imshow',
#                         'xlabel': 'Rec Location (m)',
#                         'ylabel': 'Depth (m)',
#                         'colorbar': True,
#                     }
#                 },
#                 'rel_diff': {
#                     'main': {
#                         'filt': 'transpose',
#                         'opts': {'cmap': 'gray', 'aspect': 'auto'},
#                         'title': 'Relative Difference (%)',
#                         'type': 'imshow',
#                         'xlabel': 'Rec Location (m)',
#                         'ylabel': 'Depth (m)',
#                         'colorbar': True,
#                     }
#                 },
#                 'vp_true': {
#                     'main': {
#                         'filt': 'transpose',
#                         'opts': {'cmap': 'gray', 'aspect': 'auto'},
#                         'title': '$v_{true}$',
#                         'type': 'imshow',
#                         'xlabel': 'Rec Location (m)',
#                         'ylabel': 'Depth (m)',
#                         'colorbar': True,
#                     }
#                 },
#             },
#         }
#     },
#     'train': {
#         'retrain': True,
#         'max_iters': 25,
#         'loss': {
#             'dep': {
#                 'mod': "<module 'misfit_toys.fwi.loss.tikhonov' from '/home/tyler/Documents/repos/IslandOfMisfitToys/misfit_toys/fwi/loss/tikhonov.py'>"
#             },
#             'type': "<class 'misfit_toys.fwi.loss.tikhonov.TikhonovLoss'>",
#             'builder': {
#                 'func': "<function lin_reg_drop at 0x7f3a490d0940>",
#                 'args': [],
#                 'kw': {'scale': 1e-06, '_min': 1e-07},
#             },
#         },
#         'optimizer': {
#             'type': "<function <lambda> at 0x7f3aeb6ff9a0>",
#             'builder': {
#                 'args': [],
#                 'kw': {
#                     'lr': 1.0,
#                     'max_iter': 20,
#                     'max_eval': None,
#                     'tolerance_grad': 1e-07,
#                     'tolerance_change': 1e-09,
#                     'history_size': 100,
#                     'line_search_fn': None,
#                 },
#             },
#         },
#         'step': {
#             'type': "<function taper_only at 0x7f3a4705e200>",
#             'builder': {
#                 'kw': {'length': 100, 'num_batches': None, 'scale': 1000000.0}
#             },
#         },
#         'stages': {
#             'type': "<function vanilla_stages at 0x7f3a4705e290>",
#             'builder': {'kw': {'max_iters': 25}},
#         },
#     },
#     'data': {
#         'path': '/home/tyler/anaconda3/envs/dw/data/marmousi/deepwave_example/shots16',
#         'preprocess': {
#             'dep': "<module 'misfit_toys.fwi.seismic_data' from '/home/tyler/Documents/repos/IslandOfMisfitToys/misfit_toys/fwi/seismic_data.py'>",
#             'path_builder_kw': {
#                 'remap': {'vp_init': 'vp'},
#                 'vp_init': {
#                     'type': 'self.data.dep.ParamConstrained.delay_init',
#                     'builder': {
#                         'kw': {
#                             'minv': 1000,
#                             'maxv': 2500,
#                             'requires_grad': True,
#                         }
#                     },
#                 },
#                 'src_amp_y': {
#                     'type': 'self.data.dep.Param.delay_init',
#                     'builder': {'kw': {'requires_grad': False}},
#                 },
#                 'obs_data': None,
#                 'src_loc_y': None,
#                 'src_loc_x': None,
#             },
#             'required_fields': [
#                 'vp_init',
#                 'src_amp_y',
#                 'obs_data',
#                 'src_loc_y',
#                 'src_loc_x',
#                 'meta',
#             ],
#         },
#     },
#     'rank_out': '/home/tyler/Documents/repos/IslandOfMisfitToys/misfit_toys/examples/hydra/outputs/2024-02-26/22-38-56/rank',
# }
