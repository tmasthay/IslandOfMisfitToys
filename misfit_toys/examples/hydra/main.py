import os
from time import time

import hydra
import matplotlib.pyplot as plt
import torch
import torch.multiprocessing as mp
from mh.core import DotDict, convert_dictconfig, hydra_out
from mh.core_legacy import subdict
from mh.typlotlib import apply_subplot, get_frames_bool, save_frames
from omegaconf import DictConfig
from torch.nn.parallel import DistributedDataParallel as DDP

from misfit_toys.fwi.seismic_data import SeismicProp, path_builder
from misfit_toys.fwi.training import Training
from misfit_toys.utils import (
    bool_slice,
    chunk_and_deploy,
    clean_idx,
    setup,
    apply,
    d2cpu,
    resolve,
    git_dump_info,
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
            if 'runtime_func' in v.keys():
                c.data.preprocess.path_builder_kw[k] = apply(
                    c.data.preprocess.path_builder_kw[k]
                )

    c['runtime.data'] = path_builder(
        c.data.path, **c.data.preprocess.path_builder_kw
    )

    check_keys(c, c.runtime.data)

    # Split data into chunks and deploy to GPUs
    c.runtime.data = DotDict(
        chunk_and_deploy(
            rank,
            world_size,
            data=c.runtime.data,
            chunk_keys=c.data.preprocess.chunk_keys,
        )
    )

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
    loss_fn = apply(c.train.loss)
    optimizer = apply(c.train.optimizer)
    step = apply(c.train.step)
    training_stages = apply(c.train.stages)

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
                'update': lambda x: d2cpu(x.out),
                'reduce': lambda x: torch.cat(x, dim=1),
                'presave': torch.stack,
            },
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
    c = convert_dictconfig(cfg.case)
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


def trace_plotter(*, data, idx, fig, axes, c):
    plt.clf()
    num_samples = data.out.shape[0]
    for i in range(num_samples):
        plt.subplot(*c.plt.trace.sub.shape, i + 1)
        even = 2 * i
        odd = 2 * i + 1

        ls = c.plt.trace.linestyles
        colors = c.plt.trace.color_seq
        n_ls = len(ls)
        n_colors = len(colors)
        plt.plot(
            data.obs_data[i, :],
            label='obs',
            color=colors[odd % n_colors],
            linestyle=ls[odd % n_ls],
        )
        plt.plot(
            data.out[i, idx[1], :],
            label='pred',
            color=colors[even % n_colors],
            linestyle=ls[even % n_ls],
        )
        plt.xlabel(c.plt.trace.xlabel)
        plt.ylabel(c.plt.trace.ylabel)
        plt.legend(**c.plt.trace.legend)
    plt.suptitle(f'{c.plt.trace.suptitle}\nIteration {idx[1]}')
    plt.tight_layout()
    return {'c': c}


@hydra.main(config_path="cfg", config_name="cfg", version_base=None)
def main(cfg: DictConfig) -> None:
    with open(hydra_out('git_info.txt'), 'w') as f:
        f.write(git_dump_info())

    c = preprocess_cfg(cfg)

    out_dir = os.path.join(os.path.dirname(__file__), 'out')

    def get_data():
        files = [
            os.path.join(out_dir, e)
            for e in os.listdir(out_dir)
            if e.endswith('_record.pt')
        ]
        keys = [e.replace('_record.pt', '').split('/')[-1] for e in files]
        d = DotDict({k: torch.load(f) for k, f in zip(keys, files)})
        d.obs_data = torch.load(
            os.path.join(
                c.data.path,
                "obs_data.pt",
            )
        )
        return d

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

    num_samples = c.plt.trace.sub.shape[0] * c.plt.trace.sub.shape[1]
    rand_indices = torch.stack(
        [torch.randint(0, e, (num_samples,)) for e in data.obs_data.shape[:-1]],
        dim=-1,
    )
    rand_indices = [[slice(ee, ee + 1) for ee in e] for e in rand_indices]
    traces = torch.stack([data.obs_data[s].squeeze() for s in rand_indices])
    out_traces = torch.stack(
        [data.out[[slice(None), *s]].squeeze() for s in rand_indices]
    )
    d = DotDict(
        {
            'obs_data': traces,
            'out': out_traces,
        }
    )
    trace_iter = bool_slice(*d.out.shape, **c.plt.trace.iter)
    fig, axes = plt.subplots(*c.plt.trace.sub.shape, **c.plt.trace.sub.kw)
    trace_frames = get_frames_bool(
        data=d,
        iter=trace_iter,
        fig=fig,
        axes=axes,
        plotter=trace_plotter,
        c=c,
    )
    save_frames(trace_frames, **c.plt.trace.save)

    for k, v in c.plt.items():
        print(f"Plot {k} stored in {v.save.path}")

    print('To see all output run the following in terminal:\n')
    print(f'    cd {hydra_out("")}')


# Run the script from command line
if __name__ == "__main__":
    main()
