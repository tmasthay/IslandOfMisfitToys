import os
from copy import deepcopy

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from masthay_helpers.global_helpers import DotDict, convert_config_simplest
from masthay_helpers.typlotlib import get_frames_bool, save_frames

from misfit_toys.fwi.loss.w2 import (
    W2Loss,
    W2LossConst,
    cum_trap,
    unbatch_spline_eval,
)
from misfit_toys.utils import bool_slice


def make_data(*, obs, tail_err, N, eps):
    T = -np.log(tail_err) / obs.min().item()
    # t = torch.stack([torch.linspace(-e.item(), e.item(), N) for e in T], dim=0)
    # t = torch.linspace(0.0, T, N).expand(obs.shape[0], -1)
    t = torch.linspace(0.0, T, N)
    # p = torch.linspace(0, 1, N).expand(*t.shape)
    p = torch.linspace(0, 1, N)
    # v = torch.where(obs * t >= 0, -obs * t, torch.tensor(0.0))
    v = -obs * t
    obs_pdf = torch.exp(v) + eps
    obs_pdf[obs_pdf > 1.0 - eps] = 0.0
    obs_cdf = cum_trap(obs_pdf, x=t)
    norm_constants = obs_cdf[:, -1].clone().unsqueeze(-1)
    obs_pdf /= norm_constants
    obs_cdf /= norm_constants
    tol = 1e-3
    if not (
        torch.abs(obs_cdf.max() - 1.0) < tol
        and torch.abs(obs_cdf.min()) < tol
        and all(obs_pdf.reshape(-1) >= 0)
    ):
        raise ValueError(
            'Normalization error: (max,min,positive_pdfs, nans) ='
            f' {obs_cdf.max(), obs_cdf.min(), all(obs_pdf.reshape(-1) >= 0), torch.isnan(obs_pdf).sum()}'
        )

    ref_pdf = obs_pdf[obs.shape[0] // 2].squeeze()
    ref_cdf = obs_cdf[obs.shape[0] // 2].squeeze()

    ref_pdf.requires_grad = True

    loss = W2Loss(t=t, renorm=lambda x: x, obs_data=obs_pdf, p=p)
    quantile_evaled = unbatch_spline_eval(
        loss.quantiles, p.expand(*loss.quantiles.shape, -1).clone()
    )
    quantile_evaled_deriv = unbatch_spline_eval(
        loss.quantiles, p.expand(*loss.quantiles.shape, -1).clone(), deriv=True
    )
    quantile_inverted = unbatch_spline_eval(loss.quantiles, obs_cdf)
    quantile_inverted_deriv = unbatch_spline_eval(
        loss.quantiles, obs_cdf, deriv=True
    )
    transport_map = unbatch_spline_eval(
        loss.quantiles, ref_cdf.expand(loss.quantiles.shape[-1], -1)
    )

    u = loss.batch_forward(ref_pdf.expand(*loss.quantiles.shape, -1)).detach()
    v = loss(ref_pdf.expand(*loss.quantiles.shape, -1))
    v.backward()
    grad = ref_pdf.grad
    input(grad.shape)
    return DotDict(
        {
            't': t.squeeze(),
            'p': p.squeeze(),
            'obs_pdf': obs_pdf,
            'obs_cdf': obs_cdf,
            'ref_pdf': ref_pdf.detach(),
            'ref_pdf_grad': grad.detach(),
            'ref_cdf': ref_cdf,
            # 'loss': loss,
            'quantile_evaled': quantile_evaled.squeeze(),
            'quantile_evaled_deriv': quantile_evaled_deriv.squeeze(),
            'quantile_inverted': quantile_inverted.squeeze(),
            'quantile_inverted_deriv': quantile_inverted_deriv.squeeze(),
            'transport_map': transport_map.squeeze(),
            'obs': obs,
            'ref_lambda': obs[obs.shape[0] // 2].item(),
            'loss': u,
        }
    )


def plotter(*, data, idx, fig, axes, cfg, lines=None):
    d = data
    if cfg.calls >= 0:
        for e in axes:
            for ee in e:
                ee.clear()
        lines = [[None, None], [None, None], [None], [None, None, None, None]]
        (lines[0][0],) = axes[0, 0].plot(
            d.t, d.obs_pdf[idx], label='obs_pdf', **cfg.plot.opts[0]
        )
        (lines[0][1],) = axes[0, 0].plot(
            d.t, d.ref_pdf.squeeze(), label='ref_pdf', **cfg.plot.opts[1]
        )
        axes[0, 0].legend()
        axes[0, 0].set_title(f'PDFs: idx={idx}, mean={d.obs[idx].item():.2f}')
        axes[0, 0].set_xlabel('t')
        axes[0, 0].set_ylabel('Prob. density')

        (lines[1][0],) = axes[0, 1].plot(
            d.t, d.obs_cdf[idx], label='obs_cdf', **cfg.plot.opts[0]
        )
        (lines[1][1],) = axes[0, 1].plot(
            d.t, d.ref_cdf.squeeze(), label='ref_cdf', **cfg.plot.opts[1]
        )
        axes[0, 1].legend()
        axes[0, 1].set_title('CDFs')
        axes[0, 1].set_xlabel('t')
        axes[0, 1].set_ylabel('Cumulative prob. mass')

        (lines[2][0],) = axes[1, 0].plot(
            d.p,
            d.quantile_evaled[idx],
            color='blue',
            linestyle='-',
            label='computed',
        )
        axes[1, 0].plot(
            d.p,
            -torch.log(1 - d.p) / d.obs[idx],
            color='red',
            linestyle='--',
            label='exact',
        )
        axes[1, 0].legend()
        axes[1, 0].set_title('Quantiles')
        axes[1, 0].set_xlabel('p')
        axes[1, 0].set_ylabel('t')

        quant_cutoff = cfg.nt // cfg.cutoff
        (lines[3][0],) = axes[1, 1].plot(
            d.t[:quant_cutoff],
            d.quantile_inverted[idx][:quant_cutoff],
            **cfg.plot.opts[0],
            label=r'$G^{-1}(G(t)) \approx t$',
        )
        (lines[3][1],) = axes[1, 1].plot(
            d.t[:quant_cutoff],
            d.transport_map[idx][:quant_cutoff],
            **cfg.plot.opts[1],
            label=r'$G^{-1}(F(t)) \approx \frac{\lambda_1}{\lambda_2} t$',
        )
        (lines[3][2],) = axes[1, 1].plot(
            d.t[:quant_cutoff],
            d.t[:quant_cutoff],
            **cfg.plot.opts[2],
            label=r'$I(t)= t$',
        )
        alpha = d.ref_lambda / d.obs[idx].item()
        (lines[3][3],) = axes[1, 1].plot(
            d.t[:quant_cutoff],
            d.t[:quant_cutoff] * alpha,
            label=r'$T(t) = \frac{\lambda_1}{\lambda_2} t$',
            **cfg.plot.opts[3],
        )
        axes[1, 1].legend()
        axes[1, 1].set_title('Transport maps')
        axes[1, 1].set_xlabel('t')
        axes[1, 1].set_ylabel('t')
        axes[1, 1].legend(loc='upper left', bbox_to_anchor=(1, 1))

        axes[2, 0].plot(
            d.p,
            d.quantile_evaled_deriv[idx],
            label='computed',
            **cfg.plot.opts[0],
        )
        axes[2, 0].plot(
            d.p,
            1.0 / (d.obs[idx] * (1.0 - d.p)),
            label='exact',
            **cfg.plot.opts[1],
        )
        axes[2, 0].set_title('Quantile derivative')
        axes[2, 0].set_xlabel('p')
        axes[2, 0].set_ylabel('t')
        axes[2, 0].legend()

        # ylim = d.quantile_inverted_deriv[idx].abs().max().item()
        axes[2, 1].plot(
            d.p,
            d.quantile_inverted_deriv[idx],
            label='Computed composition',
            **cfg.plot.opts[0],
        )
        axes[2, 1].plot(
            d.p,
            1.0 / (d.obs[idx] * (1.0 - d.ref_cdf)),
            label='Exact composition',
            **cfg.plot.opts[1],
        )
        # axes[2, 1].set_ylim(0.0, ylim)
        axes[2, 1].legend()
        axes[2, 1].set_title('Quantile inverse derivative')
        axes[2, 1].set_xlabel('p')
        axes[2, 1].set_ylabel('t')

        axes[3, 0].plot(
            d.obs, d.loss, **cfg.plot.opts[0], label='Computed loss'
        )
        axes[3, 0].plot(
            d.obs,
            2 * (1.0 / d.ref_lambda - 1.0 / d.obs) ** 2,
            **cfg.plot.opts[1],
            label='Exact loss',
        )
        axes[3, 0].set_title('Loss')
        axes[3, 0].set_xlabel(r'$\lambda$')
        axes[3, 0].set_ylabel('loss')
        axes[3, 0].legend()

        axes[3, 1].plot(
            d.t,
            d.ref_pdf_grad.squeeze(),
            **cfg.plot.opts[0],
            label='Computed gradient',
        )
        axes[3, 1].set_title('Gradient')
        axes[3, 1].set_xlabel('t')
        axes[3, 1].set_ylabel('dloss/dt')

    # else:
    #     # axes[0, 0].set_data(d.t, d.obs_pdf[idx])
    #     # axes[0, 1].set_data(d.t, d.obs_cdf[idx])
    #     # axes[1, 0].set_data(d.p, d.quantile_evaled[idx])
    #     # quant_cutoff = cfg.nt // cfg.cutoff
    #     # axes[1, 1].set_data(
    #     #     d.t[:quant_cutoff],
    #     #     d.quantile_inverted[:quant_cutoff],
    #     # )
    #     lines[0][0].set_ydata(d.obs_pdf[idx])
    #     lines[1][0].set_ydata(d.obs_cdf[idx])
    #     lines[2][0].set_ydata(d.quantile_evaled[idx])
    #     quant_cutoff = cfg.nt // cfg.cutoff
    #     lines[3][0].set_data(
    #         d.t[:quant_cutoff], d.quantile_inverted[idx][:quant_cutoff]
    #     )
    #     lines[3][1].set_data(
    #         d.t[:quant_cutoff], d.transport_map[idx][:quant_cutoff]
    #     )
    #     lines[3][2].set_data(d.t[:quant_cutoff], d.t[:quant_cutoff])
    #     lines[3][3].set_data(
    #         d.t[:quant_cutoff], d.t[:quant_cutoff] * d.obs[idx] / d.ref_lambda
    #     )

    cfg.calls += 1

    return {'cfg': cfg, 'lines': lines}


@hydra.main(config_path='config', config_name='simple', version_base=None)
def main(cfg):
    cfg = convert_config_simplest(cfg)
    obs = torch.linspace(*cfg.exp_params).unsqueeze(-1)

    cfg.ref_idx = cfg.exp_params[-1] // 2
    d = make_data(obs=obs, tail_err=cfg.tail_err, N=cfg.nt, eps=cfg.eps)

    fig, axes = plt.subplots(*cfg.plot.shape, figsize=cfg.plot.figsize)
    plt.subplots_adjust(
        hspace=cfg.plot.wspace, wspace=cfg.plot.hspace, right=cfg.plot.right
    )

    indices = bool_slice(
        *d.obs_pdf.shape, none_dims=(1,), ctrl=(lambda x, y: True)
    )
    kw = deepcopy(cfg)
    kw.calls = 0
    frames = get_frames_bool(
        data=d, iter=indices, fig=fig, axes=axes, plotter=plotter, cfg=kw
    )
    save_frames(frames, path=cfg.plot.name, duration=cfg.plot.duration)


if __name__ == '__main__':
    main()
