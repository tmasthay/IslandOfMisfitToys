import hydra
import matplotlib.pyplot as plt
import torch
from mh.core import DotDict, convert_dictconfig
from mh.typlotlib import get_frames_bool, save_frames
from omegaconf import DictConfig
from returns.curry import curry
from torch.nn import MSELoss

from misfit_toys.fwi.loss.w2 import (
    W2Loss,
    cts_quantile,
    cum_trap,
    unbatch_spline_eval,
)
from misfit_toys.utils import bool_slice, clean_idx, pull_data, tensor_summary


def plotter(*, data=None, idx, fig, axes, cfg):
    def rec_extremes(*args):
        args = list(args)
        for i in range(len(args)):
            args[i] = args[i][idx[0], :, :]
        min_val = min([x.min().item() for x in args])
        max_val = max([x.max().item() for x in args])
        return min_val, max_val

    def lplot(title, xlabel, ylabel, cbar=True, yext=None):
        plt.title(f'{title} idx={clean_idx(idx)}')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if cfg.plt.static_ylim and yext is not None:
            plt.ylim(*yext)
        if cbar:
            plt.colorbar()

    def set_plot(i):
        plt.subplot(*cfg.plt.subplot.shape, cfg.plt.order[i - 1])

    def leg(i):
        plt.legend(**cfg.plt.legend[cfg.plt.order[i - 1] % len(cfg.plt.legend)])

    plt.clf()
    set_plot(1)
    plt.imshow(
        cfg.true.rescaled_obs[idx[0], :, :].T,
        **cfg.plt.imshow,
        vmin=cfg.true.rescaled_obs[idx[0], :, :].min().item(),
        vmax=cfg.true.rescaled_obs[idx[0], :, :].max().item(),
    )
    plt.plot(
        idx[1] * torch.ones(cfg.meta.nt),
        torch.arange(cfg.meta.nt),
        **cfg.plt.trace[3],
    )
    plt.plot(idx[1], cfg.meta.nt - 1, 'r*')
    lplot('Depth-scaled obs data', 'Receiver Index', 'Time Index')

    set_plot(2)
    plt.imshow(
        cfg.init.rescaled_obs[idx[0], :, :].T,
        **cfg.plt.imshow,
        vmin=cfg.init.rescaled_obs[idx[0], :, :].min().item(),
        vmax=cfg.init.rescaled_obs[idx[0], :, :].max().item(),
    )
    plt.plot(
        idx[1] * torch.ones(cfg.meta.nt),
        torch.arange(cfg.meta.nt),
        **cfg.plt.trace[3],
    )
    lplot('Depth-scaled init data', 'Receiver Index', 'Time Index')

    set_plot(3)
    plt.plot(
        cfg.t,
        cfg.true.obs_data[idx],
        **cfg.plt.trace[0],
        label=r'$\mathcal{R}^{-1}(g)$',
    )
    plt.plot(
        cfg.t,
        cfg.init.obs_data[idx],
        **cfg.plt.trace[1],
        label=r'$\mathcal{R}^{-1}(f)$',
    )
    leg(3)
    lplot(
        'Raw obs data trace',
        'Time',
        'Amplitude',
        cbar=False,
        yext=rec_extremes(cfg.true.obs_data, cfg.init.obs_data),
    )

    set_plot(4)
    plt.plot(
        cfg.t, cfg.true.obs_data_renorm[idx], **cfg.plt.trace[0], label=r'$g$'
    )
    plt.plot(
        cfg.t, cfg.init.obs_data_renorm[idx], **cfg.plt.trace[1], label=r'$f$'
    )
    leg(4)
    lplot(
        'Renormed data trace',
        'Time',
        'Amplitude',
        cbar=False,
        yext=rec_extremes(cfg.true.obs_data_renorm, cfg.init.obs_data_renorm),
    )

    set_plot(5)
    plt.plot(
        cfg.t, cfg.true.obs_data_cdf[idx], **cfg.plt.trace[0], label=r'$G(t)$'
    )
    plt.plot(
        cfg.t, cfg.init.obs_data_cdf[idx], **cfg.plt.trace[1], label=r'$F(t)$'
    )
    leg(5)
    lplot(
        'CDF trace',
        't',
        'p',
        cbar=False,
        yext=rec_extremes(cfg.true.obs_data_cdf, cfg.init.obs_data_cdf),
    )

    set_plot(6)
    plt.plot(
        cfg.p, cfg.quantiles[idx], **cfg.plt.trace[0], label=r'$G^{-1}(p)$'
    )
    plt.plot(
        cfg.p, cfg.init.quantiles[idx], **cfg.plt.trace[1], label=r'$F^{-1}(p)$'
    )
    leg(6)
    lplot('Quantiles', 'p', 't', cbar=False, yext=rec_extremes(cfg.quantiles))

    set_plot(7)
    plt.plot(
        cfg.t,
        cfg.transport_maps[idx],
        **cfg.plt.trace[0],
        label=r'$T(t) = G^{-1}(F(t))$',
    )
    plt.plot(cfg.t, cfg.t, **cfg.plt.trace[1], label=r'$I(t) = t$')
    lplot(
        'Transport maps',
        't',
        't',
        cbar=False,
        yext=(cfg.t.min().item(), cfg.t.max().item()),
    )
    leg(7)

    set_plot(8)
    w2_normed = cfg.w2_losses[idx[0], :] / cfg.w2_losses[idx[0], :].max()
    l2_normed = cfg.l2_losses[idx[0], :] / cfg.l2_losses[idx[0], :].max()
    plt.plot(
        range(cfg.meta.rec_per_shot),
        w2_normed,
        **cfg.plt.trace[0],
        label=r'Relative $W_2^2(f,g)$',
    )
    plt.plot(
        range(cfg.meta.rec_per_shot),
        l2_normed,
        **cfg.plt.trace[1],
        label=r'Relative $L_2^2$ loss',
    )
    plt.plot(
        idx[1],
        w2_normed[idx[1]],
        *cfg.plt.marker[0].args,
        **cfg.plt.marker[0].kwargs,
    )
    plt.plot(
        idx[1],
        l2_normed[idx[1]],
        *cfg.plt.marker[1].args,
        **cfg.plt.marker[1].kwargs,
    )
    leg(8)
    lplot(r'$W_2^2$ loss', 'Receiver Index', r'$W_2^2$', cbar=False)

    set_plot(9)
    plt.imshow(cfg.init.vp_true, **cfg.plt.vp.imshow)
    lplot('Initial $v_p$', 'x', 'z', cbar=True)

    set_plot(10)
    plt.imshow(cfg.true.vp_init, **cfg.plt.vp.imshow)
    lplot('True $v_p$', 'x', 'z', cbar=True)

    set_plot(11)
    plt.imshow(cfg.w2_gradients[idx[0], :, :].T, **cfg.plt.vp.imshow)
    lplot(r'$W_2$ gradients', 'Receiver Index', 'Time Index', cbar=True)

    set_plot(12)
    plt.plot(cfg.t, cfg.w2_gradients[idx], **cfg.plt.trace[0])
    lplot(r'$W_2$ gradients', 't', 'gradient', cbar=False)

    set_plot(13)
    plt.imshow(cfg.l2_gradients[idx[0], :, :].T, **cfg.plt.vp.imshow)
    lplot(r'$L_2$ gradients', 'Receiver Index', 'Time Index', cbar=True)

    set_plot(14)
    plt.plot(cfg.t, cfg.l2_gradients[idx], **cfg.plt.trace[0])
    lplot(r'$L_2$ gradients', 't', 'gradient', cbar=False)

    return {'cfg': cfg}


def convert_cfg(cfg):
    cfg = convert_dictconfig(cfg)
    cfg.t = torch.linspace(0.0, cfg.meta.nt * cfg.meta.dt, cfg.meta.nt)
    cfg.p = torch.linspace(0.0, 1.0, cfg.meta.nt)
    cfg.plt.order = cfg.plt.order or range(
        1, 1 + cfg.plt.subplot.shape[0] * cfg.plt.subplot.shape[1]
    )
    cfg.t_scaled = (1.0 + cfg.t**cfg.plt.depth_scaling).expand(1, 1, -1)
    return cfg


@hydra.main(config_path='conf', config_name='main', version_base=None)
def main(cfg: DictConfig) -> None:
    cfg = convert_cfg(cfg)
    cfg.true = pull_data(cfg.data.path.true)
    cfg.init = pull_data(cfg.data.path.init)
    cfg.true.rescaled_obs = cfg.true.obs_data * cfg.t_scaled
    cfg.init.rescaled_obs = cfg.init.obs_data * cfg.t_scaled

    input(torch.unique(cfg.true.vp_init))
    input(torch.unique(cfg.init.vp_true))

    # print('TRUE')
    # for k, v in true.items():
    #     print(f'{k}\n{tensor_summary(v)}')
    # print('INIT')
    # for k, v in init.items():
    #     print(f'{k}\n{tensor_summary(v)}')

    def renorm(x):
        u = torch.abs(x) + cfg.preproc.renorm_perturb
        c = torch.trapz(u, x=cfg.t, dim=-1).unsqueeze(-1)
        return u / c

    cfg.true.obs_data_renorm = renorm(cfg.true.obs_data)
    cfg.init.obs_data_renorm = renorm(cfg.init.obs_data)
    cfg.true.obs_data_cdf = cum_trap(cfg.true.obs_data_renorm, x=cfg.t, dim=-1)
    cfg.init.obs_data_cdf = cum_trap(cfg.init.obs_data_renorm, x=cfg.t, dim=-1)
    init_quantile_cts = cts_quantile(cfg.init.obs_data_renorm, cfg.t, cfg.p)
    cfg.init.quantiles = unbatch_spline_eval(
        init_quantile_cts, cfg.p.expand(*init_quantile_cts.shape, -1)
    )

    cfg.loss_fn = W2Loss(
        renorm=renorm, p=cfg.p, t=cfg.t, obs_data=cfg.true.obs_data
    )
    cfg.quantiles = unbatch_spline_eval(
        cfg.loss_fn.quantiles, cfg.p.expand(*cfg.loss_fn.quantiles.shape, -1)
    )
    cfg.transport_maps = unbatch_spline_eval(
        cfg.loss_fn.quantiles, cfg.init.obs_data_cdf
    )

    cfg.w2_losses = cfg.loss_fn.batch_forward(cfg.init.obs_data)
    cfg.l2_losses = torch.sum(
        (cfg.true.obs_data - cfg.init.obs_data) ** 2, dim=-1
    )

    cfg.init.obs_data.requires_grad = True
    u = cfg.loss_fn(cfg.init.obs_data)
    u.backward()
    cfg.w2_gradients = cfg.init.obs_data.grad

    tmp = cfg.init.obs_data.detach().clone()
    tmp.requires_grad = True
    my_loss = MSELoss()
    u = my_loss(tmp, cfg.true.obs_data)
    u.backward()
    cfg.l2_gradients = tmp.grad

    cfg.init.obs_data.requires_grad = False

    fig, axes = plt.subplots(*cfg.plt.subplot.shape, **cfg.plt.subplot.kwargs)
    plt.subplots_adjust(**cfg.plt.subplot.adjust)
    iter = bool_slice(
        *cfg.true.obs_data.shape,
        none_dims=cfg.plt.sel.none_dims,
        ctrl=(lambda x, y: True),
        start=cfg.plt.sel.start,
        cut=cfg.plt.sel.cut,
        strides=cfg.plt.sel.strides,
    )
    frames = get_frames_bool(
        data=None, iter=iter, fig=fig, axes=axes, cfg=cfg, plotter=plotter
    )
    save_frames(frames, path=cfg.plt.path, duration=cfg.plt.duration)


if __name__ == '__main__':
    main()
