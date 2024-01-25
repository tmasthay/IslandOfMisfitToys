import os
from collections import OrderedDict

import hydra
import torch
import torch.multiprocessing as mp
from masthay_helpers.global_helpers import DotDict, clean_kwargs, subdict
from masthay_helpers.typlotlib import (
    get_frames_bool,
    make_gifs,
    save_frames,
    slice_iter_bool,
)
from matplotlib import pyplot as plt
from returns.curry import curry
from scipy.signal import butter
from torch.nn.parallel import DistributedDataParallel as DDP

from misfit_toys.fwi.loss.tikhonov import TikhonovLoss
from misfit_toys.fwi.loss.w2 import W2LossConst, cum_trap, unbatch_spline_eval
from misfit_toys.fwi.seismic_data import (
    Param,
    ParamConstrained,
    SeismicProp,
    chunk_and_deploy,
    path_builder,
)
from misfit_toys.fwi.training import Training
from misfit_toys.utils import bool_slice, filt, setup, taper


@curry
def my_renorm(x, dt):
    u = torch.abs(x)
    return u / cum_trap(u, dx=dt, dim=-1)[-1].to(u.device)


def plotter(*, data, idx, fig, axes, t, p, star=None):
    print(f'idx = {idx}')
    x_star, y_star = idx[0], idx[1]

    if idx[0] + idx[1] == 0:
        (star,) = axes[0, 0].plot(x_star, y_star, 'r*')
        axes[0, 0].set_ylim(-1, data.quantiles.shape[1] + 1)
        axes[0, 0].set_xlim(-1, data.quantiles.shape[0] + 1)
        axes[0, 0].set_title('Experiment Reference')
        axes[0, 0].set_xlabel('Shot Number')
        axes[0, 0].set_ylabel('Receiver Number')

    else:
        star.set_data(x_star, y_star)

    axes[0, 1].cla()
    axes[0, 1].plot(t, data.obs_data[idx])
    axes[0, 1].set_title('Raw obs data')
    axes[0, 1].set_xlabel('t')
    axes[0, 1].set_ylabel('Amplitude')

    axes[1, 0].cla()
    axes[1, 0].plot(t, data.renorm_obs_data[idx])
    axes[1, 0].set_title('Renormalized obs data')
    axes[1, 0].set_xlabel('t')
    axes[1, 0].set_ylabel('Amplitude')

    axes[1, 1].cla()
    axes[1, 1].plot(p, data.quantiles[idx])
    axes[1, 1].set_title(
        f'Trace Renormalized Quantile: (shot_no, rec_no) = ({idx[0], idx[1]})'
    )
    axes[1, 1].set_xlabel('p')
    axes[1, 1].set_ylabel('t')

    axes[2, 0].cla()
    axes[2, 0].plot(p, data.shifted_up_pdf[idx])
    axes[2, 0].set_title(
        f'Trace Renormalized PDF: (shot_no, rec_no) = ({idx[0], idx[1]})'
    )
    axes[2, 0].set_xlabel('t')
    axes[2, 0].set_ylabel('p')

    axes[2, 1].cla()
    axes[2, 1].plot(p, data.cumulative[idx])
    axes[2, 1].set_title(
        f'Trace Renormalized Cumulative: (shot_no, rec_no) = ({idx[0], idx[1]})'
    )
    axes[2, 1].set_xlabel('t')
    axes[2, 1].set_ylabel('p')

    return {'star': star, 't': t, 'p': p}


# Main function for training on each rank
def run(cfg):
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

    nt, dt = data['meta'].nt, data['meta'].dt

    t = torch.linspace(0, (nt - 1) * dt, nt)
    p = torch.linspace(0, 1, nt)

    # preprocess data like Alan and then deploy slices onto GPUs
    data["obs_data"] = taper(data["obs_data"])

    eps = 1e-5
    v = torch.abs(data['obs_data']) + eps
    v /= torch.trapz(v, dx=dt, dim=-1).unsqueeze(-1)

    input(v.shape)
    input(t.shape)
    input(p.shape)
    loss_fn = W2LossConst(t=t, renorm=(lambda x: x), obs_data=v, p=p)
    input(loss_fn.quantiles.shape)

    u = unbatch_spline_eval(
        loss_fn.quantiles, p.expand(*loss_fn.quantiles.shape, -1)
    )
    cdf = cum_trap(v, dx=dt, dim=-1)

    # if cdf.max() - 1.0 > eps:
    #     raise ValueError(f'max(cdf) = {cdf.max()}, should be near 1.0')

    # slice_dims = (2, 20, u.shape[-1])
    slice_dims = u.shape if not cfg.plot.debug else cfg.plot.debug
    slicer = bool_slice(*slice_dims, none_dims=(2,), ctrl=(lambda x, y: True))
    fig, axes = plt.subplots(3, 2, figsize=(20, 10))
    plt.subplots_adjust(wspace=0.5, hspace=0.5)

    frames = get_frames_bool(
        data=DotDict(
            {
                'quantiles': u,
                'obs_data': data['obs_data'],
                'renorm_obs_data': loss_fn.renorm_obs_data,
                'cumulative': cdf,
                'shifted_up_pdf': v,
            }
        ),
        iter=slicer,
        fig=fig,
        axes=axes,
        plotter=plotter,
        t=torch.linspace(0, 1, 300),
        p=torch.linspace(0, 1, 300),
    )
    save_frames(frames, path='quantiles.gif', duration=cfg.plot.common.duration)


# Main function for running the script
@hydra.main(config_path='.', config_name='config', version_base=None)
def main(cfg):
    if cfg.exec.run:
        run(cfg)
        # plot_data(cfg)


# Run the script from command line
if __name__ == "__main__":
    main()
