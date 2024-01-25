import os
import pickle
from functools import wraps
from time import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from gen_data import gen_data
from masthay_helpers.global_helpers import DotDict
from masthay_helpers.typlotlib import (
    get_frames_bool,
    save_frames,
    slice_iter_bool,
)
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from returns.curry import curry

from misfit_toys.fwi.loss.w2 import cts_quantile, cum_trap, unbatch_spline_eval
from misfit_toys.utils import bool_slice

# def omit_none_slices(func):
#     @wraps(func)
#     def wrapper(x):
#         # Filter out 'slice(None)' from x
#         x = [e for e in x if e != slice(None)]
#         return func(x)

#     return wrapper


def plotter(
    *, data: DotDict, idx, fig: Figure, axes: Axes, line=None, x, p, ylim=None
):
    if ylim is None:
        ylim = (x.min(), x.max())
    # plot_call_start = time()
    x_star, y_star = idx[0], idx[1]

    if idx[0] + idx[1] == 0:
        (line,) = axes[0, 0].plot(x_star, y_star, 'r*')
        axes[0, 0].set_ylim(-1, data.gauss.shape[1] + 1)
        axes[0, 0].set_xlim(-1, data.gauss.shape[0] + 1)
        axes[0, 0].set_xlabel('Standard deviation')
        axes[0, 0].set_ylabel('Mean')
        axes[0, 0].set_title('Current mean and standard deviation')
    else:
        line.set_data([x_star], [y_star])

    axes[0, 1].cla()
    try:
        axes[0, 1].plot(p, data.quantile_clean[idx], label=r'$Q(p)$')
    except Exception:
        input(f'idx = {idx}')
        input(f'p.shape = {p.shape}')
        input(f'data.quantile_clean.shape = {data.quantile_clean.shape}')
        raise
    axes[0, 1].set_title('Quantile Function on uniform grid')
    axes[0, 1].set_xlabel('p')
    axes[0, 1].set_ylabel('t')

    axes[1, 0].cla()
    axes[1, 0].plot(x, data.ref, label=r'PDF $f(x)$ (reference)')
    axes[1, 0].plot(x, data.gauss[idx], label=r'PDF $g(x)$ (shifted & scaled)')
    axes[1, 0].set_title('PDF')
    axes[1, 0].set_xlabel('t')
    axes[1, 0].set_ylabel('p')
    axes[1, 0].legend()

    axes[1, 1].cla()
    axes[1, 1].plot(x, data.cum_ref.squeeze(), label=r'CDF $F(t)$ (reference)')
    axes[1, 1].plot(
        x, data.cum_gauss[idx], label=r'CDF $G(t)$ (shifted & scaled)'
    )
    axes[1, 1].set_title('CDF')
    axes[1, 1].set_xlabel('t')
    axes[1, 1].set_ylabel('p')
    axes[1, 1].legend()

    axes[2, 0].cla()
    axes[2, 0].plot(x, data.approx_iden[idx], label=r'$G^{-1}(G(t)) \approx t$')
    axes[2, 0].plot(x, x, label=r'$I(x) := x$')
    axes[2, 0].set_xlabel('t')
    axes[2, 0].set_ylabel('Predicted t')
    axes[2, 0].set_title('Approximate identity')
    axes[2, 0].legend()

    axes[2, 1].cla()
    axes[2, 1].plot(
        x, data.transport_maps[idx], label=r'$T(t) := G^{-1}(F(t))$'
    )
    axes[2, 1].plot(x, x, label=r'$I(x) := x$')
    axes[2, 1].plot(x, data.transport_maps[idx] - x, label=r'$T(t) - I(x)$')
    axes[2, 1].set_xlabel('t')
    axes[2, 1].set_ylabel('Predicted t')
    axes[2, 1].set_title('Transport map')
    axes[2, 1].legend()

    return {'line': line, 'x': x, 'ylim': ylim, 'p': p}


def plot_data(*, data, name, x, p, duration):
    def ctrl(idx, shape):
        return True

    iter = bool_slice(
        *data.gauss.shape,
        none_dims=(2,),
        ctrl=ctrl,
    )
    fig, axes = plt.subplots(3, 2, figsize=(15, 10))
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    frames = get_frames_bool(
        data=data,
        iter=iter,
        fig=fig,
        axes=axes,
        plotter=plotter,
        x=x,
        p=p,
    )
    save_frames(
        frames, path=f'{name.replace(".gif", "")}.gif', duration=duration
    )


def main():
    u = gen_data(
        config_path=os.path.dirname(__file__), config_name='config.yaml'
    )
    u = u.gauss
    # u.gauss = u.gauss[:3, :3]
    quantiles = cts_quantile(u.gauss, u.x, u.p, dx=u.x[1] - u.x[0])
    cum_ref = cum_trap(
        u.ref, dx=u.x[1] - u.x[0], dim=-1, preserve_dims=True
    ).reshape(1, 1, -1)
    cum_gauss = cum_trap(
        u.gauss, dx=u.x[1] - u.x[0], dim=-1, preserve_dims=True
    )

    tol = 1e-1
    max_ref = cum_ref.max()
    min_ref = cum_ref.min()
    max_gauss = cum_gauss.max()
    min_gauss = cum_gauss.min()

    if torch.abs(max_ref - 1.0) >= tol:
        raise ValueError(f'max_ref = {max_ref}, should be near 1.0')
    assert torch.abs(min_ref) < tol
    assert torch.abs(max_gauss - 1.0) < tol
    assert torch.abs(min_gauss) < tol

    approx_iden = unbatch_spline_eval(quantiles, cum_gauss)
    transport_maps = unbatch_spline_eval(
        quantiles, cum_ref.expand(quantiles.shape[0], quantiles.shape[1], -1)
    )
    quantile_clean = unbatch_spline_eval(
        quantiles, u.p.expand(*quantiles.shape, -1)
    )

    data = DotDict(
        {
            'ref': u.ref,
            'gauss': u.gauss,
            'cum_ref': cum_ref,
            'cum_gauss': cum_gauss,
            'approx_iden': approx_iden,
            'transport_maps': transport_maps,
            'quantile_clean': quantile_clean,
        }
    )
    plot_data(data=data, name='test', x=u.x, p=u.p, duration=1000)


if __name__ == '__main__':
    main()
