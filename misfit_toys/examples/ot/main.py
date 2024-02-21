import os
from collections import OrderedDict

import hydra
import torch
import torch.multiprocessing as mp
from mh.core import clean_kwargs
from mh.core_legacy import subdict
from mh.typlotlib import make_gifs
from returns.curry import curry
from scipy.signal import butter
from torch.nn import MSELoss
from torch.nn.parallel import DistributedDataParallel as DDP

from misfit_toys.fwi.loss.tikhonov import TikhonovLoss
from misfit_toys.fwi.loss.w2 import cum_trap
from misfit_toys.fwi.seismic_data import (
    Param,
    ParamConstrained,
    SeismicProp,
    path_builder,
)
from misfit_toys.fwi.training import Training
from misfit_toys.utils import (
    chunk_and_deploy,
    filt,
    setup,
    taper,
    tensor_summary,
)


def training_stages(cfg):
    def do_nothing(x, y):
        return None

    return {
        'epochs': {
            'data': list(range(cfg.exec.epochs)),
            'preprocess': do_nothing,
            'postprocess': do_nothing,
        }
    }


# Define _step for the training class
def _step(self):
    self.out = self.prop(1)
    self.loss = 1e6 * self.loss_fn(self.out[-1], self.obs_data)
    if self.loss.item() < 0.0:
        raise ValueError(
            'Negative loss encountered:'
            f' {self.loss.item()}\n'
            f'obs_data\n{tensor_summary(self.obs_data)}\n'
            f'out\n{tensor_summary(self.out[-1])}'
            f'\nrenorm_out\n{tensor_summary(self.loss_fn.renorm(self.out[-1]))}\n'
            f' renorm_obs_data\n{tensor_summary(self.loss_fn.renorm(self.obs_data))}'
        )
    # print(f'self.loss = {self.loss}', flush=True)
    # raise ValueError(
    #     f'self.loss = {self.loss}, self.out.min() = {self.out[-1].min()},'
    #     f' self.out.max() = {self.out[-1].max()}, self.out.mean() ='
    #     f' {self.out[-1].mean()}, self.out.std() = {self.out[-1].std()}'
    # )
    self.loss.backward()
    return self.loss


# Syntactic sugar for converting from device to cpu
def d2cpu(x):
    return x.detach().cpu()


# Define regularization decay functions
def reg_decay(key=None):
    @clean_kwargs
    @curry
    def alpha_linear(iter, max_iters, *, _min=0.0, _max=0.01):
        if iter > max_iters:
            return _min
        return _max + (_min - _max) * iter / max_iters

    @clean_kwargs
    @curry
    def alpha_exp(iter, max_iters, *, _min=0.0, _max=0.01, beta=1.0):
        if iter > max_iters or _max == 0.0:
            return _min
        return _max * (_min / _max) ** (beta * iter / max_iters)

    @clean_kwargs
    @curry
    def const(iter, max_iters, *, _min=0.0, _max=0.01):
        return _max

    options = {'linear': alpha_linear, 'exp': alpha_exp, 'const': const}
    return options if key is None else options[key]


def get_loss_fn(cfg, **kw):
    loss_cfg = cfg.exec.loss
    options = {'tik': loss_cfg.tik, 'w2': loss_cfg.w2}
    if loss_cfg.type not in options.keys():
        return torch.nn.MSELoss()

    chosen = options[loss_cfg.type]

    if loss_cfg.type == 'tik':
        return TikhonovLoss(
            weights=kw['prop'].module.vp,
            alpha=reg_decay(chosen.alpha.function)(**chosen.alpha.kw),
            max_iters=chosen.alpha.max_iters,
        )
    elif loss_cfg.type == 'w2':
        # device = kw['obs_data'].device
        # return W2LossConst(
        #     t=kw['t'].to(device),
        #     renorm=kw['renorm'],
        #     obs_data=kw['obs_data'],
        #     p=kw['p'].to(device),
        # )
        return MSELoss()
    else:
        raise NotImplementedError(f'Loss type {loss_cfg.type} not implemented')


# Main function for training on each rank
def run_rank(rank, world_size, cfg):
    print(f"Running DDP on rank {rank} / {world_size}.")
    setup(rank, world_size)

    # Build data for marmousi model
    data = path_builder(
        cfg.exec.data_path,
        remap=cfg.exec.remap,
        vp_init=ParamConstrained.delay_init(
            minv=cfg.exec.min_vel, maxv=cfg.exec.max_vel, requires_grad=True
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

    # Build seismic propagation module and wrap in DDP
    prop_data = subdict(data, exc=["obs_data"])
    prop = SeismicProp(
        **prop_data,
        max_vel=cfg.exec.max_vel,
        pml_freq=data["meta"].freq,
        time_pad_frac=cfg.exec.time_pad_frac,
    ).to(rank)
    prop = DDP(prop, device_ids=[rank])

    # def my_renorm(x):
    #     u = x**2 + 1e-3
    #     return u / cum_trap(u, dx=data['meta'].dt, dim=-1)[-1].to(u.device)

    def my_renorm2(x):
        u = x**2 + 1e-3
        c = torch.trapz(u, dx=data['meta'].dt, dim=-1)
        if torch.count_nonzero(c) != c.numel():
            raise ValueError(f'Zero integral encountered: {tensor_summary(c)}')
        return u / c.unsqueeze(-1)

    def my_renorm(x):
        u = torch.abs(x) + 1e-3
        c = torch.trapz(u, dx=data['meta'].dt, dim=-1).unsqueeze(-1)
        if torch.count_nonzero(c) != c.numel():
            raise ValueError(f'Zero integral encountered: {tensor_summary(c)}')
        return u / c

    # raise ValueError(
    #     f'{tensor_summary(data["obs_data"])}\n{tensor_summary(my_renorm(data["obs_data"]))}'
    # )

    used_loss_fn = get_loss_fn(
        cfg,
        prop=prop,
        t=torch.linspace(0, 1.196, 300),
        p=torch.linspace(0, 1, 300),
        renorm=my_renorm,
        obs_data=data["obs_data"],
    )

    used_loss_fn = MSELoss()

    train = Training(
        rank=rank,
        world_size=world_size,
        prop=prop,
        obs_data=data["obs_data"],
        loss_fn=used_loss_fn,
        optimizer=[torch.optim.LBFGS, {}],
        verbose=1,
        report_spec={
            'path': os.path.join(os.path.dirname(__file__), cfg.io.tensor),
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
            # 'out': {
            #     'update': lambda x: d2cpu(x.out[-1]),
            #     'reduce': lambda x: torch.cat(x, dim=1),
            #     'presave': torch.stack,
            # },
            # 'out_filt': {
            #     'update': lambda x: d2cpu(x.out_filt),
            #     'reduce': lambda x: torch.cat(x, dim=1),
            #     'presave': torch.stack,
            # },
        },
        _step=_step,
        _build_training_stages=(lambda: training_stages(cfg)),
    )
    train.train()


# Main function for spawning ranks
def run(world_size, cfg):
    mp.spawn(run_rank, args=(world_size, cfg), nprocs=world_size, join=True)


def plot_data(cfg):
    in_dir = os.path.relpath(cfg.io.tensor, start=os.path.dirname(__file__))
    common = {**cfg.plot.common, 'path': os.path.join(in_dir, cfg.io.figs)}
    opts = {
        'loss_record': {'labels': ['Epoch', 'Loss']},
        'vp_record': {
            'labels': ['Extent', 'Depth', 'Epoch'],
            'permute': (2, 1, 0),
        },
        # 'out_record': {
        #     'labels': ['Extent', 'Time', 'Shot No', 'Epoch'],
        #     'permute': (3, 2, 1, 0),
        # },
    }
    # opts['out_filt_record'] = opts['out_record']
    for k in opts.keys():
        opts[k].update(common)

    make_gifs(
        in_dir=in_dir,
        out_dir=common['path'],
        targets=list(opts.keys()),
        opts=opts,
        cmap='seismic',
    )


# Main function for running the script
@hydra.main(config_path='.', config_name='config', version_base=None)
def main(cfg):
    if cfg.exec.run:
        n_gpus = torch.cuda.device_count()
        run(min(cfg.exec.n_gpus, n_gpus), cfg)
    plot_data(cfg)


# Run the script from command line
if __name__ == "__main__":
    main()
