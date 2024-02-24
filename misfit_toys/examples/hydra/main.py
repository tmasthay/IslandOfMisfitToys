import os
from collections import OrderedDict

import torch
import torch.multiprocessing as mp
from mh.core_legacy import subdict
from scipy.signal import butter
from torch.nn.parallel import DistributedDataParallel as DDP

from misfit_toys.fwi.seismic_data import (
    Param,
    ParamConstrained,
    SeismicProp,
    path_builder,
)
from misfit_toys.fwi.training import Training
from misfit_toys.utils import chunk_and_deploy, filt, setup, taper, clean_idx
from helpers import W2Loss
from misfit_toys.fwi.loss.tikhonov import TikhonovLoss, lin_reg_tmp
import hydra
from omegaconf import DictConfig
from mh.typlotlib import get_frames_bool, save_frames, apply_subplot
from misfit_toys.utils import bool_slice
from mh.core import DotDict, convert_dictconfig, hydra_out, exec_imports
import matplotlib.pyplot as plt
from time import time

torch.set_printoptions(precision=3, sci_mode=True, threshold=5, linewidth=10)


def apply_legacy(lcl, gbl):
    chosen = lcl[lcl.chosen.lower()]
    if 'chosen' not in chosen.keys():
        if 'type' not in chosen.keys():
            raise ValueError(
                f'Expected type key in {chosen} since no chosen key. Consider restructuring config.'
            )
        obj = chosen.type(*chosen.args, **chosen.kw)
        return obj
    sub_chosen = chosen[chosen.chosen.lower()]
    args = sub_chosen.get('args', [])
    kwargs = sub_chosen.get('kw', {}) or sub_chosen.get('kwargs', {})
    obj = sub_chosen.func(gbl, *args, **kwargs)
    if 'type' in chosen.keys():
        if type(obj) == tuple:
            args2, kwargs2 = obj
        else:
            args2 = obj['args']
            if 'kw' not in obj.keys() and 'kwargs' not in obj.keys():
                raise ValueError(f'Need kw or kwargs in {obj}')
            kwargs2 = obj.get('kw', {}) or obj.get('kwargs', {})
        obj = chosen.type(*args2, **kwargs2)
    return obj


def apply(lcl, gbl):
    builder = lcl.builder
    print(builder, flush=True)
    if 'func' in builder.keys():
        args, kwargs = builder.func(gbl, *builder.args, **builder.kw)
    else:
        args = builder.get('args', [])
        kwargs = builder.get('kw', {}) or builder.get('kwargs', {})
    obj = lcl.type(*args, **kwargs)
    return obj


def training_stages(c):
    def helper():
        def do_nothing(training, epoch):
            pass

        return {
            'epoch': {
                'data': range(c.train.max_iters),
                'preprocess': do_nothing,
                'postprocess': do_nothing,
            }
        }

    return helper


# Define _step for the training class
def _step(self):
    self.out = self.prop(1)

    # IGNORE LINE BELOW FOR NOW -- THIS IS FREQ-FILTERING FROM ALAN'S CODE
    # self.out_filt = filt(taper(self.out[-1]), self.sos)

    # FOR TIKHONOV
    #     UNCOMMENT BLOCK BELOW
    #     COMMENT OUT THE "FOR W2" BLOCK BELOW
    self.out_filt = taper(self.out[-1])
    self.obs_data_filt = taper(self.obs_data)
    self.loss = 1e6 * self.loss_fn(self.out_filt, self.obs_data_filt)

    # FOR W2
    #    COMMENT OUT EVERYTHING IN THE BLOCK ABOVE
    #    AND UNCOMMENT THE BELOW
    # self.loss = 1.0e6 * self.loss_fn(self.out[-1])
    self.loss.backward()
    return self.loss


# Syntactic sugar for converting from device to cpu
def d2cpu(x):
    return x.detach().cpu()


# Main function for training on each rank
def run_rank(rank: int, world_size: int, c: DotDict) -> None:
    print(f"Running DDP on rank {rank} / {world_size}.")
    setup(rank, world_size)

    start_pre = time()
    c = resolve(c, relax=True)
    print(f"Preprocessing took {time() - start_pre:.2f} seconds.", flush=True)
    # Build data for marmousi model
    # raise ValueError(c.data.path)
    data = path_builder(
        c.data.path,
        remap={"vp_init": "vp"},
        vp_init=ParamConstrained.delay_init(
            minv=1000, maxv=5000, requires_grad=True
        ),
        src_amp_y=Param.delay_init(requires_grad=False),
        obs_data=None,
        src_loc_y=None,
        rec_loc_y=None,
    )

    # preprocess data like Alan and then deploy slices onto GPUs
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

    if torch.isnan(data['vp'].p).any():
        raise ValueError("NaNs in vp")

    # Build seismic propagation module and wrap in DDP
    prop_data = subdict(data, exc=["obs_data"])
    c.obs_data = data["obs_data"]
    c.prop = SeismicProp(
        **prop_data, max_vel=2500, pml_freq=data["meta"].freq, time_pad_frac=0.2
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
        verbose=2,
        report_spec={
            'path': os.path.join(os.path.dirname(__file__), 'out'),
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
        _step=_step,
        _build_training_stages=training_stages(c),
    )
    train.train()


# Main function for spawning ranks
def run(world_size: int, c: DotDict) -> None:
    mp.spawn(run_rank, args=(world_size, c), nprocs=world_size, join=True)


def preprocess_cfg(cfg: DictConfig) -> DotDict:
    c = convert_dictconfig(cfg)
    for k, v in c.plt.items():
        c[f'plt.{k}.save.path'] = hydra_out(v.save.path)
    c.data.path = c.data.path.replace('conda', os.environ['CONDA_PREFIX'])

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

    iter = bool_slice(*data.vp.shape, **c.plt.vp.iter)
    fig, axes = plt.subplots(*c.plt.vp.sub.shape, **c.plt.vp.sub.kw)
    if c.plt.vp.sub.adjust:
        plt.subplots_adjust(**c.plt.vp.sub.adjust)
    data.vp = data.vp.permute(0, 2, 1)
    vp_true = torch.load(
        os.path.join(
            os.environ["CONDA_PREFIX"],
            "data/marmousi/deepwave_example/shots16",
            "vp_true.pt",
        )
    )
    c.vp_true = vp_true.T.unsqueeze(0)
    c.rel_diff = data.vp - c.vp_true
    c.rel_diff = c.rel_diff / torch.abs(c.vp_true) * 100.0
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


# Run the script from command line
if __name__ == "__main__":
    main()
