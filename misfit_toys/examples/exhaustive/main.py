import hydra
from omegaconf import DictConfig
from masthay_helpers.global_helpers import DotDict, convert_config_simplest
from misfit_toys.utils import bool_slice, clean_idx, pull_data, tensor_summary
from masthay_helpers.typlotlib import get_frames_bool, save_frames
import matplotlib.pyplot as plt
from returns.curry import curry
from misfit_toys.fwi.loss.w2 import (
    W2LossConst,
    cum_trap,
    unbatch_spline_eval,
    cts_quantile,
)
import torch


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

    plt.clf()
    set_plot(1)
    plt.imshow(
        cfg.true.rescaled_obs[idx[0], :, :].T,
        **cfg.plt.imshow,
    )
    plt.plot(idx[1], cfg.meta.rec_per_shot // 2, 'r*')
    lplot('Depth-scaled obs data', 'Receiver Index', 'Time Index')

    set_plot(2)
    plt.imshow(cfg.init.rescaled_obs[idx[0], :, :].T, **cfg.plt.imshow)
    plt.plot(idx[1], cfg.meta.rec_per_shot // 2, 'r*')
    lplot('Depth-scaled init data', 'Receiver Index', 'Time Index')

    set_plot(3)
    plt.plot(cfg.t, cfg.true.obs_data[idx], **cfg.plt.trace[0])
    plt.plot(cfg.t, cfg.init.obs_data[idx], **cfg.plt.trace[1])
    lplot(
        'Raw obs data trace',
        'Time',
        'Amplitude',
        cbar=False,
        yext=rec_extremes(cfg.true.obs_data, cfg.init.obs_data),
    )

    set_plot(4)
    plt.plot(cfg.t, cfg.true.obs_data_renorm[idx], **cfg.plt.trace[0])
    plt.plot(cfg.t, cfg.init.obs_data_renorm[idx], **cfg.plt.trace[1])
    lplot(
        'Renormed data trace',
        'Time',
        'Amplitude',
        cbar=False,
        yext=rec_extremes(cfg.true.obs_data_renorm, cfg.init.obs_data_renorm),
    )

    set_plot(5)
    plt.plot(cfg.t, cfg.true.obs_data_cdf[idx], **cfg.plt.trace[0])
    plt.plot(cfg.t, cfg.init.obs_data_cdf[idx], **cfg.plt.trace[1])
    lplot(
        'CDF trace',
        't',
        'p',
        cbar=False,
        yext=rec_extremes(cfg.true.obs_data_cdf, cfg.init.obs_data_cdf),
    )

    set_plot(6)
    plt.plot(cfg.p, cfg.quantiles[idx], **cfg.plt.trace[0])
    plt.plot(cfg.p, cfg.init.quantiles[idx], **cfg.plt.trace[1])
    lplot(
        'Quantiles',
        'p',
        't',
        cbar=False,
        yext=rec_extremes(cfg.quantiles),
    )

    return {'cfg': cfg}


def convert_cfg(cfg):
    cfg = convert_config_simplest(cfg)
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

    cfg.loss_fn = W2LossConst(
        renorm=renorm, p=cfg.p, t=cfg.t, obs_data=cfg.true.obs_data
    )
    cfg.quantiles = unbatch_spline_eval(
        cfg.loss_fn.quantiles, cfg.p.expand(*cfg.loss_fn.quantiles.shape, -1)
    )

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
