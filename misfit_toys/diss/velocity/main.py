# @VS@ cd _dir && python -W ignore _file +run=test

import os
from os.path import exists as exists
from os.path import join as pj
from subprocess import check_output as co
from time import time

import hydra
import matplotlib.pyplot as plt
import torch
import torch.multiprocessing as mp
import yaml
from mh.core import (
    DotDict,
    convert_dictconfig,
    hydra_out,
    set_print_options,
    torch_stats,
)
from mh.core_legacy import subdict
from mh.typlotlib import apply_subplot, get_frames_bool, save_frames
from omegaconf import DictConfig
from torch.nn.parallel import DistributedDataParallel as DDP

from misfit_toys.fwi.seismic_data import DebugProp, SeismicProp, path_builder
from misfit_toys.fwi.training import Training
from misfit_toys.swiffer import dupe
from misfit_toys.utils import (
    apply,
    bool_slice,
    chunk_and_deploy,
    clean_idx,
    cleanup,
    d2cpu,
    git_dump_info,
    resolve,
    runtime_reduce,
    setup,
)


def set_options():
    opts = ['shape']
    set_print_options(
        precision=3,
        sci_mode=True,
        threshold=5,
        linewidth=10,
        callback=torch_stats(opts),
    )
    return


set_options()

# def hydra_out(name: str = '') -> str:
#     out = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
#     s = os.path.join(out, name)
#     base_folder = os.path.dirname(s)
#     os.makedirs(base_folder, exist_ok=True)
#     return s


def sco(cmd, verbose=False):
    cmd = ' '.join(cmd.split())
    if verbose:
        print(cmd, flush=True)
    return co(cmd, shell=True).decode().strip()


def check_keys(c, data):
    """
    Check if all the required fields are present in the data dictionary.

    Args:
        c (object): The configuration object.
        data (dict): The data dictionary to be checked.

    Raises:
        ValueError: If any of the required fields are missing in the data dictionary.

    """
    pre = c.data.preprocess
    required = pre.required_fields
    if 'remap' in pre.path_builder_kw.keys():
        for k, v in pre.path_builder_kw.remap.items():
            if k in required:
                required.remove(k)
                required.append(v)
    if not set(required).issubset(data.keys()):
        raise ValueError(
            "Missing required fields in data\n "
            f"    Required: {pre.required_fields}\n"
            f"    Found: {list(data.keys())}"
        )


def finished_writing(*, names, world_size, path):
    def rank_name(rank, name):
        return pj(path, name, f"_{rank}.pt")

    for rank in world_size:
        for name in names:
            if not exists(rank_name(rank, name)):
                return False
    return True


def run_rank(rank: int, world_size: int, c: DotDict) -> None:
    """
    Runs the DDP (Distributed Data Parallel) training on a specific rank.

    Args:
        rank (int): The rank of the current process.
        world_size (int): The total number of processes.
        c (DotDict): The configuration dictionary.

    Returns:
        None
    """
    print(f"Running DDP on rank {rank} / {world_size}.")
    setup(rank, world_size, port=c.port)

    if c.get('dupe', False):
        dupe(hydra_out('stream'), editor=c.get('editor', None))

    start_pre = time()
    c = resolve(c, relax=True)
    print(f"Preprocessing took {time() - start_pre:.2f} seconds.", flush=True)

    for k, v in c.data.preprocess.path_builder_kw.items():
        if isinstance(v, dict) or isinstance(v, DotDict):
            if 'runtime_func' in v.keys():
                c.data.preprocess.path_builder_kw[k] = apply(
                    c.data.preprocess.path_builder_kw[k]
                )

    c.data.preprocess.path_builder_kw = runtime_reduce(
        c.data.preprocess.path_builder_kw
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
    c.runtime.t = torch.linspace(
        0.0,
        c.runtime.data.meta.dt * c.runtime.data.meta.nt,
        c.runtime.data.meta.nt,
    ).to(rank)

    # Build seismic propagation module and wrap in DDP
    # prop_data = subdict(c.runtime.data, exc=["obs_data"])
    # c.obs_data = c.runtime.data.obs_data
    # print(prop_data.keys(), flush=True)

    # keys:
    # - vp
    # - meta
    # - src_loc_y
    # - rec_loc_y
    # - src_amp_y
    # c['runtime.prop'] = SeismicProp(
    #     **prop_data,
    #     max_vel=c.data.preprocess.maxv,
    #     pml_freq=c.runtime.data.meta.freq,
    #     time_pad_frac=c.data.preprocess.time_pad_frac,
    # ).to(rank)
    # c['runtime.prop'] = DebugProp(
    #     vp=prop_data['vp'],
    #     dx=prop_data['meta']['dx'],
    #     dt=prop_data['meta']['dt'],
    #     freq=prop_data['meta']['freq'],
    #     rec_loc_y=prop_data['rec_loc_y'],
    #     src_loc_y=prop_data['src_loc_y']
    # )
    c = resolve(c, relax=True)

    c.runtime.prop = apply(c.train.prop).to(rank)
    c.runtime.prop = DDP(c.runtime.prop, device_ids=[rank])

    c = resolve(c, relax=False)

    loss_fn = apply(c.train.loss)
    if hasattr(loss_fn, 'to'):
        loss_fn = loss_fn.to(rank)

    optimizer = apply(c.train.optimizer)
    step = apply(c.train.step)
    training_stages = apply(c.train.stages)

    pre_time = time() - start_pre
    print(f"Preprocess time rank {rank}: {pre_time:.2f} seconds.", flush=True)

    def _post_train(self):
        self._save_report()

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
        _post_train=_post_train,
    )
    train_start = time()
    train.train()
    train_time = time() - train_start
    print(f"Train time rank {rank}: {train_time:.2f} seconds.", flush=True)

    # torch.distributed.barrier()
    # print('Past barrier', flush=True)
    # cleanup()
    # make multiprocessing barrier
    # mp.barrier()


# Main function for spawning ranks
def run(world_size: int, c: DotDict) -> None:
    """
    Runs the main worker process.

    Args:
        world_size (int): The number of processes to spawn.
        c (DotDict): The configuration for the worker process.

    Returns:
        None
    """
    mp.spawn(run_rank, args=(world_size, c), nprocs=world_size, join=True)


def preprocess_cfg(cfg: DictConfig) -> DotDict:
    """
    Preprocesses the configuration dictionary.

    Args:
        cfg (DictConfig): The input configuration dictionary.

    Returns:
        DotDict: The preprocessed configuration dictionary.

    Raises:
        ValueError: If 'name' is not provided in the configuration dictionary.

    """
    if 'run' not in cfg.keys():
        raise ValueError(
            "Name must be provided at command line with"
            " +run=NAME_DESCRIBING_THIS_RUN.\nThis protocol is for disciplined"
            " tracking of output files."
        )
    c = convert_dictconfig(cfg.case)
    dump_resolved_config = convert_dictconfig(
        cfg, self_ref_resolve=False, mutable=False
    )
    resolved_config_str = yaml.dump(dump_resolved_config.__dict__)
    resolved_config_str = resolved_config_str.replace(
        '!!python/object:mh.core.DotDict', ''
    )
    with open(hydra_out('resolved_config.yaml'), 'w') as f:
        f.write(resolved_config_str)
    del dump_resolved_config
    for k, v in c.plt.items():
        c[f'plt.{k}.save.path'] = hydra_out(v.save.path)
    c.data.path = c.data.path.replace('conda', os.environ['CONDA_PREFIX'])
    c.rank_out = hydra_out(c.get('rank_out', 'rank'))

    # later on if you want to delay this execution, that is an easy refactor
    #     just add a key that says to do so or not.
    # c.data.preprocess.addons.self_ref_resolve()
    if 'addons' in c.data.preprocess.keys():
        c.data.preprocess.addons = resolve(
            c.data.preprocess.addons, relax=False
        )
        c.data.preprocess.addons = apply(c.data.preprocess.addons)

    c.data.postprocess = resolve(c.data.postprocess, relax=False)
    # c.data.postprocess = apply(c.data.postprocess)

    return c


def plotter(*, data, idx, fig, axes, c):
    """
    Plots the data and returns a dictionary containing the input parameters.

    Args:
        data (numpy.ndarray): The input data.
        idx (numpy.ndarray): The index of the data to plot.
        fig (matplotlib.figure.Figure): The figure object to plot on.
        axes (matplotlib.axes.Axes): The axes object to plot on.
        c (dict): A dictionary containing configuration parameters.

    Returns:
        dict: A dictionary containing the input parameters.

    """
    # plt.imshow(data[idx], **c.plt.vp)
    plt.clf()
    vp_true = c.vp_true.squeeze()

    extent = [0.0, 4.0 * vp_true.shape[0], vp_true.shape[1] * 4.0, 0.0]
    lim_rel = {
        'vmin': c.rel_diff.min(),
        'vmax': c.rel_diff.max(),
        'extent': extent,
    }
    lim_vp = {'vmin': vp_true.min(), 'vmax': vp_true.max(), 'extent': extent}
    apply_subplot(
        data=data[idx], cfg=c.plt.vp, name='vp', layer='main', **lim_vp
    )
    apply_subplot(
        data=c.rel_diff[idx],
        cfg=c.plt.vp,
        name='rel_diff',
        layer='main',
        **lim_rel,
    )
    apply_subplot(
        data=c.vp_true.squeeze(),
        cfg=c.plt.vp,
        name='vp_true',
        layer='main',
        **lim_vp,
    )
    plt.subplot(*c.plt.vp.sub.shape, 4)
    plt.plot(c.loss)
    plt.scatter(idx[0], c.loss[idx[0]], color='r', s=100, marker='o')
    plt.title('Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.suptitle(f'Iteration {idx[0]}')
    plt.tight_layout()
    return {'c': c}


def trace_plotter(*, data, idx, fig, axes, c):
    """
    Plots the observed and predicted data for each sample in a trace plot.

    Args:
        data (numpy.ndarray): The input data.
        idx (tuple): The index of the iteration.
        fig (matplotlib.figure.Figure): The figure object to plot on.
        axes (matplotlib.axes.Axes): The axes object to plot on.
        c (dict): The configuration parameters for the plot.

    Returns:
        dict: A dictionary containing the configuration parameters.

    """
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


@hydra.main(config_path="all/main", config_name="default", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main function for processing data and generating plots.

    Args:
        cfg (DictConfig): Configuration object containing the settings.

    Returns:
        None
    """

    input(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    with open(hydra_out('git_info.txt'), 'w') as f:
        f.write(git_dump_info())

    c = preprocess_cfg(cfg)

    out_dir = os.path.join(os.path.dirname(__file__), 'out')
    os.makedirs(out_dir, exist_ok=True)

    def get_data():
        files = [
            os.path.join(out_dir, e)
            for e in os.listdir(out_dir)
            if e.endswith('_record.pt')
        ]
        keys = [e.replace('_record.pt', '').split('/')[-1] for e in files]
        d = DotDict({k: torch.load(f) for k, f in zip(keys, files)})
        d.obs_data = torch.load(os.path.join(c.data.path, "obs_data.pt"))
        return d

    # this is inefficient, but it was a quick fix to a bug from a long time ago.
    #     A simple if condition should make reading data twice unnecessary.
    data = get_data()
    train_time = 0.0
    if not data or c.train.retrain:
        training_start = time()
        n_gpus = torch.cuda.device_count()
        run(n_gpus, c)
        train_time = time() - training_start
        print(f"Training alone took {train_time:.2f} seconds.")
        data = get_data()

    vp_true = torch.load(os.path.join(c.data.path, "vp_true.pt"))
    data.vp_true = vp_true

    pp_kw = c.data.postprocess.get('kw', {})
    c.data.postprocess.__call__(
        data, path=hydra_out(), train_time=train_time, **pp_kw
    )

    c.plt = resolve(c.plt, relax=False)
    iter = bool_slice(*data.vp.shape, **c.plt.vp.iter)
    fig, axes = plt.subplots(*c.plt.vp.sub.shape, **c.plt.vp.sub.kw)
    if c.plt.vp.sub.adjust:
        plt.subplots_adjust(**c.plt.vp.sub.adjust)

    c.vp_true = vp_true.unsqueeze(0)
    c.rel_diff = data.vp - c.vp_true
    c.rel_diff = c.rel_diff / torch.abs(c.vp_true) * 100.0
    c.loss = data.loss
    frames = get_frames_bool(
        data=data.vp, iter=iter, fig=fig, axes=axes, plotter=plotter, c=c
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
    d = DotDict({'obs_data': traces, 'out': out_traces})
    trace_iter = bool_slice(*d.out.shape, **c.plt.trace.iter)
    fig, axes = plt.subplots(*c.plt.trace.sub.shape, **c.plt.trace.sub.kw)
    trace_frames = get_frames_bool(
        data=d, iter=trace_iter, fig=fig, axes=axes, plotter=trace_plotter, c=c
    )
    save_frames(trace_frames, **c.plt.trace.save)

    for k, v in c.plt.items():
        print(f"Plot {k} stored in {v.save.path}")

    print('To see all output run the following in terminal:\n')
    print(f'    cd {hydra_out("")}')

    print('Equivalently, you can run the following command:\n')
    print('    . .latest_run')

    with open(os.path.join(os.getcwd(), '.latest_run'), 'w') as f:
        f.write(f'cd {hydra_out("")}')
        # set permissions to execute
        os.chmod(os.path.join(os.getcwd(), '.latest_run'), 0o755)


# Run the script from command line
if __name__ == "__main__":
    set_options()
    main()
