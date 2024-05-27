import matplotlib.pyplot as plt
import torch
from mh.core import DotDict
from mh.typlotlib import apply_subplot, get_frames_bool, save_frames

from misfit_toys.utils import bool_slice


def plotter(*, data, idx, fig, axes, c):
    plt.clf()
    curr_plot = c.plt.gm.plts.loss.main
    curr_data = data.loss_history.detach().cpu()
    apply_subplot(data=curr_data, cfg=c.plt.gm, name='loss', layer='main')
    plt.plot(
        [idx[0]], [curr_data[idx[0]]], *curr_plot.dot.args, **curr_plot.dot.kw
    )

    curr_plot = c.plt.gm.plts.grad_norm.main
    curr_data = data.grad_norm_history.detach().cpu()
    apply_subplot(data=curr_data, cfg=c.plt.gm, name='grad_norm', layer='main')
    plt.plot(
        [idx[0]], [curr_data[idx[0]]], *curr_plot.dot.args, **curr_plot.dot.kw
    )

    apply_subplot(
        data=data.soln_history.detach().cpu()[idx],
        cfg=c.plt.gm,
        name='soln',
        layer='curr',
    )
    apply_subplot(
        data=c.rt.gm.obs_data.detach().cpu(),
        cfg=c.plt.gm,
        name='soln',
        layer='obs_data',
    )

    apply_subplot(
        data=data.grad_history.detach().cpu()[idx],
        cfg=c.plt.gm,
        name='grad',
        layer='curr',
    )

    plt.tight_layout()
    return {'c': c}


def monitor(c: DotDict):
    fig, axes = plt.subplots(*c.plt.gm.sub.shape, **c.plt.gm.sub.kw)
    iter = bool_slice(*c.rt.gm.train.soln_history.shape, **c.plt.gm.iter)
    frames = get_frames_bool(
        data=c.rt.gm.train, iter=iter, fig=fig, axes=axes, plotter=plotter, c=c
    )
    save_frames(frames, **c.plt.gm.save)
    print(f'Plots saved to {c.plt.gm.save.path}')
