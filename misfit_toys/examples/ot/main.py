import os

import torch
import torch.multiprocessing as mp
from scipy.signal import butter
from collections import OrderedDict
from masthay_helpers.global_helpers import subdict, clean_kwargs
from misfit_toys.utils import setup, filt, taper
from misfit_toys.fwi.training import Training

from torch.nn.parallel import DistributedDataParallel as DDP
from misfit_toys.fwi.seismic_data import (
    SeismicProp,
    Param,
    ParamConstrained,
    path_builder,
    chunk_and_deploy,
)
from misfit_toys.fwi.loss.w2 import W2Loss, str_to_renorm
from misfit_toys.fwi.loss.tikhonov import TikhonovLoss
from returns.curry import curry
from masthay_helpers.typlotlib import make_gifs
import hydra


def training_stages(cfg):
    do_nothing = lambda x, y: None
    return {
        'epochs': {
            'data': list(range(cfg.exec.epochs)),
            'preprocess': do_nothing,
            'postprocess': do_nothing,
        }
    }
    # # define training stages for the training class
    # def freq_preprocess(training, freq):
    #     # sos = butter(6, freq, fs=1 / training.prop.module.meta.dt, output="sos")
    #     # sos = [torch.tensor(sosi).to(training.obs_data.dtype) for sosi in sos]

    #     # training.sos = sos

    #     # training.obs_data_filt = filt(training.obs_data, sos)

    #     training.reset_optimizer()

    # def freq_postprocess(training, freq):
    #     pass

    # def epoch_preprocess(training, epoch):
    #     pass

    # def epoch_postprocess(training, epoch):
    #     pass

    # return OrderedDict(
    #     [
    #         (
    #             "freqs",
    #             {
    #                 "data": range(1),
    #                 "preprocess": freq_preprocess,
    #                 "postprocess": freq_postprocess,
    #             },
    #         ),
    #         (
    #             "epochs",
    #             {
    #                 "data": list(range(20)),
    #                 "preprocess": epoch_preprocess,
    #                 "postprocess": epoch_postprocess,
    #             },
    #         ),
    #     ]
    # )


# Define _step for the training class
def _step(self):
    self.out = self.prop(1)
    # self.out_filt = filt(taper(self.out[-1]), self.sos)
    # self.loss = 1e6 * self.loss_fn(self.out_filt, self.obs_data_filt)
    self.loss = 1e6 * self.loss_fn(self.out[-1], self.obs_data)
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
    options = {
        'tik': loss_cfg.tik,
    }
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
        return W2Loss(R=str_to_renorm(chosen.renorm))
    else:
        raise NotImplementedError(f'Loss type {loss_cfg.type} not implemented')


# Main function for training on each rank
def run_rank(rank, world_size, cfg):
    print(f"Running DDP on rank {rank} / {world_size}.")
    setup(rank, world_size)

    # Build data for marmousi model
    data = path_builder(
        cfg.exec.data_path,
        remap=cfg.exec.remap,  # remap should be designed better? -> data read in doesn't have to be standardized which is nice but it makes the workflow a bit more confusing
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

    train = Training(
        rank=rank,
        world_size=world_size,
        prop=prop,
        obs_data=data["obs_data"],
        loss_fn=get_loss_fn(cfg, prop=prop),
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
    # torch.distributed.barrier()


# Main function for spawning ranks
def run(world_size, cfg):
    mp.spawn(run_rank, args=(world_size, cfg), nprocs=world_size, join=True)


def plot_data(cfg):
    in_dir = os.path.relpath(cfg.io.tensor, start=os.path.dirname(__file__))
    common = {**cfg.plot.common, 'path': os.path.join(in_dir, cfg.io.figs)}
    opts = {
        'loss_record': {
            'labels': ['Epoch', 'Loss'],
        },
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
        run(n_gpus, cfg)
    plot_data(cfg)
    # torch.distributed.barrier()


# Run the script from command line
if __name__ == "__main__":
    main()
