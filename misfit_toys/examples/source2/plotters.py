import matplotlib.pyplot as plt
from mh.typlotlib import get_frames_bool, save_frames

from misfit_toys.utils import bool_slice, clean_idx


def iter_sugar(*, data_shape, shape=None, **kw):
    if shape is None:
        shape = data_shape
    return bool_slice(*shape, **kw)


def simple_imshow(
    *, data, imshow, title, save_path=None, xlabel=None, ylabel=None
):
    plt.clf()
    if not imshow.transpose:
        plt.imshow(data, **imshow.kw)
    else:
        plt.imshow(data.T, **imshow.kw)
    if imshow.colorbar:
        plt.colorbar()
    if imshow.legend:
        plt.legend()
    plt.title(title)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if save_path is not None:
        plt.savefig(save_path)


def easy_plot(
    *,
    data,
    iter,
    plotter,
    subplot_shape=None,
    subplot_kw=None,
    framer=None,
    path,
    movie_format='gif',
    duration=100,
    verbose=False,
    loop=0,
):
    subplot_shape = subplot_shape or (1, 1)
    subplot_kw = subplot_kw or {}
    fig, axes = plt.subplots(*subplot_shape, **subplot_kw)

    final_iter = iter_sugar(data_shape=data.shape, **iter)
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


def plot_gbl_obs_data(
    *,
    data,
    iter,
    imshow,
    title='Observed Data',
    path='obs_data',
    loop=0,
    duration=100,
    verbose=False,
    movie_format='gif',
    subplot_shape=None,
    subplot_kw=None,
):
    def plotter(*, data, idx, fig, axes):
        simple_imshow(
            data=data[idx],
            imshow=imshow,
            title=f'{title}: {clean_idx(idx)}',
            save_path=None,
        )

    easy_plot(
        data=data,
        iter=iter,
        plotter=plotter,
        subplot_shape=subplot_shape,
        subplot_kw=subplot_kw,
        path=path,
        movie_format=movie_format,
        duration=duration,
        verbose=verbose,
        loop=loop,
    )
