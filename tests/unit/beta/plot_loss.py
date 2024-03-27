import os

import matplotlib.pyplot as plt
import torch
from mh.core import hydra_out
from mh.typlotlib import get_frames_bool, save_frames

from misfit_toys.utils import bool_slice, clean_idx


def verify_and_plot(*, plotter, d):
    make_plots = plotter(d)
    pass_loss = d.loss_history[-1] <= d.atol
    pass_rtol = d.loss_history[-1] <= d.rtol * d.loss_history[0]
    pass_mse = d.mse_history[-1] <= d.atol
    pass_rtol_mse = d.mse_history[-1] <= d.rtol * d.mse_history[0]
    if pass_loss and pass_rtol and pass_mse and pass_rtol_mse:
        make_plots('SUCCESS')
    else:
        make_plots('FAILURE')
        assert pass_loss, f'loss = {d.loss_history[-1]} > {d.atol}'
        assert (
            pass_rtol
        ), f'rel_loss = {d.loss_history[-1] / d.loss_history[0]} > {d.rtol}'
        assert pass_mse, f'mse = {d.mse_history[-1]} > {d.atol}'
        assert (
            pass_rtol_mse
        ), f'rel_mse = {d.mse_history[-1] / d.mse_history[0]} > {d.rtol}'
    return True


def should_plot(*, status, name, max_plots, out_path):
    path = os.path.join(out_path, 'beta/loss/figs')
    os.makedirs(path, exist_ok=True)
    already_plotted = [
        e for e in os.listdir(path) if e.startswith(name) and e.endswith('.jpg')
    ]
    path = f'{path}/{name}_{status}'
    if len(already_plotted) >= max_plots or os.path.exists(path):
        return ''
    return path


def plotter_loss(*, data, idx, fig, axes):
    plt.clf()
    plt.suptitle(f'Iteration: {clean_idx(idx)}')
    plt.subplot(*data.plot.sub.shape, 1)
    plt.plot(data.t, data.soln_history[idx], label='Prediction')
    plt.plot(data.t, data.data, label='Ground truth', linestyle='--')
    plt.title('Solution')

    plt.subplot(*data.plot.sub.shape, 2)
    plt.plot(data.t, data.grad_history[idx])
    plt.title('Gradient')

    plt.subplot(*data.plot.sub.shape, 3)
    plt.plot(range(len(data.loss_history)), data.loss_history)
    plt.plot([idx[0]], [data.loss_history[idx[0]]], 'ro')
    plt.title('Loss')

    plt.subplot(*data.plot.sub.shape, 4)
    plt.plot(range(len(data.mse_history)), data.mse_history)
    plt.plot([idx[0]], [data.mse_history[idx[0]]], 'ro')
    plt.title('MSE')

    if 'int_plotter' in data.plot:
        data.plot.int_plotter(data=data, idx=idx, fig=fig, axes=axes)
    plt.tight_layout()


def plot_loss(d):
    def helper(status):
        path = should_plot(
            status=status,
            name=d.name,
            max_plots=d.plot.max_plots,
            out_path=d.out_path,
        )
        if path:
            fig, axes = plt.subplots(*d.plot.sub.shape, **d.plot.sub.kw)

            iter = bool_slice(*d.soln_history.shape, **d.plot.iter)
            frames = get_frames_bool(
                data=d, iter=iter, fig=fig, axes=axes, plotter=plotter_loss
            )
            save_frames(frames, path=path, **d.plot.save)
            print(f'Plots saved in {path}', flush=True)
        return path

    return helper
