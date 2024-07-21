import os
from os.path import join as pj

import deepwave as dw
import matplotlib.pyplot as plt
import torch
from deepwave.wavelets import ricker
from mh.typlotlib import get_frames_bool, save_frames

from misfit_toys.types import FlatPlotter
from misfit_toys.utils import apply_all, bool_slice


def create_velocity_model(
    *, ny, nx, default, piecewise_boxes, smoother, device
):
    v = default * torch.ones(ny, nx)

    def rel2abs(coord, n):
        if type(coord) == int:
            assert 0 <= coord < n, f"coord={coord} is not in bounds [0, {n})"
            return coord
        elif type(coord) == float:
            assert (
                0 <= coord <= 1
            ), f"Relative coord={coord} is not in bounds [0, 1] with n={n}"
            return int(coord * n)
        else:
            raise ValueError(f"coord must be int or float, not {type(coord)}")

    if piecewise_boxes is not None:
        for box in piecewise_boxes:
            y_left, y_right, x_left, x_right, value = box
            y_left = rel2abs(y_left, ny)
            y_right = rel2abs(y_right, ny)
            x_left = rel2abs(x_left, nx)
            x_right = rel2abs(x_right, nx)
            v[y_left:y_right, x_left:x_right] = value
    if smoother is not None:
        v = smoother(v.unsqueeze(0))
    return v.squeeze().to(device)


def plot_vp(*, data, imshow, title, save_path):
    plt.clf()
    plt.imshow(data, **imshow.kw)
    if imshow.colorbar:
        plt.colorbar()
    if imshow.legend:
        plt.legend()
    plt.title(title)
    plt.savefig(save_path)


def plot_src_loc_idx(*, data, idx, imshow, title):
    plt.clf()
    plt.plot(data[idx], **imshow.kw)
    if imshow.colorbar:
        plt.colorbar()
    if imshow.legend:
        plt.legend()
    plt.title(title)


def iter_sugar(*, data_shape, shape=None, **kw):
    if shape is None:
        shape = data_shape
    return bool_slice(*shape, **kw)


def easy_plot(
    *,
    data,
    iter,
    plotter,
    subplot_shape,
    subplot_kw,
    framer=None,
    path,
    movie_format='gif',
    duration=100,
    verbose=False,
    loop=0,
):
    fig, axes = plt.subplots(*subplot_shape, **subplot_kw)

    final_iter = iter_sugar(data_shape=data.shape, iter=iter)
    frames = get_frames_bool(
        data=data,
        iter=final_iter,
        fig=fig,
        axes=axes,
        plotter=plotter,
        framer=framer,
    )
    save_frames(
        frames,
        path=path,
        movie_format=movie_format,
        duration=duration,
        verbose=verbose,
        loop=loop,
    )


def gen_src_loc(*, n_shots, src_per_shot, ndims, depth, d_src, fst_src, device):
    assert src_per_shot == 1, "Only 1 source per shot is supported for now"
    src_loc = torch.zeros(
        n_shots, src_per_shot, ndims, device=device, dtype=torch.long
    )
    src_loc[..., 1] = depth
    src_loc[:, 0, 0] = torch.arange(n_shots) * d_src + fst_src
    return src_loc


def gen_rec_loc(*, n_shots, rec_per_shot, ndims, depth, d_rec, fst_rec, device):
    rec_loc = torch.zeros(
        n_shots, rec_per_shot, ndims, device=device, dtype=torch.long
    )
    rec_loc[..., 1] = depth
    rec_loc[:, :, 0] = (torch.arange(n_shots) * d_rec + fst_rec).repeat(
        n_shots, 1
    )
    return rec_loc


def gen_time_sig(*, freq, nt, dt, peak_time):
    return ricker(freq, nt, dt, peak_time)


def same_src_amp(
    *, n_shots, src_per_shot, nt, dt, peak_time_factor, freq, device
):
    time_sig = gen_time_sig(
        freq=freq, nt=nt, dt=dt, peak_time=peak_time_factor / freq
    )
    return time_sig.repeat(n_shots, src_per_shot, 1).to(device)


def gen_obs_data(**kw):
    return dw.scalar(**kw)[-1]
