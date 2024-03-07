import numpy as np
from scipy.interpolate import splrep, BSpline
from torchcubicspline import NaturalCubicSpline, natural_cubic_spline_coeffs
import torch
import matplotlib.pyplot as plt
from masthay_helpers.typlotlib import get_frames_bool, save_frames
from misfit_toys.utils import bool_slice, mean_filter_1d as mf
from masthay_helpers.global_helpers import DotDict, convert_config_correct
import hydra
import os
from misfit_toys.fwi.loss.w2 import cum_trap, quantile, spline_func


@hydra.main(config_path='cfg', config_name='tmp', version_base=None)
def main(c):
    c = convert_config_correct(c)
    if c.sub.order is None:
        c.sub.order = range(1, c.sub.shape[0] * c.sub.shape[1] + 1)

    d = DotDict({})
    files = [e[:-3] for e in os.listdir(c.path) if e.endswith('.pt')]
    for f in files:
        d.set(f, torch.load(f'{c.path}/{f}.pt'))
    d.meta = DotDict(eval(open(f'{c.path}/metadata.pydict', 'r').read()))
    d.t = torch.linspace(0, (d.meta.nt - 1) * d.meta.dt, d.meta.nt)
    d.t_pert = torch.sort(d.t + c.eta_t * torch.rand(d.t.shape)).values
    d.rdata = torch.abs(d.obs_data)
    d.rdata /= torch.trapz(d.rdata, dx=d.meta.dt, axis=-1).unsqueeze(-1)
    d.p = torch.linspace(0.0, 1.0, c.np)
    d.cdf = cum_trap(d.rdata, d.t)
    d.cdf = d.cdf / d.cdf[..., -1].unsqueeze(-1)
    d.q = quantile(d.rdata, d.t, d.p, ltol=c.ltol, rtol=c.rtol)
    d.q_deriv_exact = d.q(d.p, deriv=True)
    d.q_deriv_smooth = mf(d.q_deriv_exact, c.window_size)
    d.q_exact = d.q(d.p)
    # d.q_deriv_smooth = spline_func(d.p, d.q_deriv_smooth)

    tnp, tp = d.t.numpy(), d.t_pert.numpy()

    def set_plot(i):
        plt.subplot(*c.sub.shape, c.sub.order[i - 1])

    def get_opts(i, j):
        return c.sub.opts[c.sub.order[i - 1] - 1][j - 1]

    def plotter(*, data, idx, fig, axes):
        # ynp = d.cdf[idx].squeeze().numpy()
        # spl = BSpline(*splrep(tnp, ynp, s=0))(tp)
        # spl_s = BSpline(*splrep(tnp, ynp, s=c.s))(tp)

        raw = d.obs_data[idx].squeeze()
        pdf = d.rdata[idx].squeeze()
        cdf = d.cdf[idx].squeeze()
        q = d.q_exact[idx].squeeze()
        qd = d.q_deriv_exact[idx].squeeze()
        qds = d.q_deriv_smooth[idx].squeeze()
        q_rec_smooth = cum_trap(d.q_deriv_smooth[idx].squeeze(), d.p) + q[0]
        q_rec_exact = cum_trap(d.q_deriv_exact[idx].squeeze(), d.p) + q[0]
        # q_rec_smooth_err = torch.abs(q_rec_smooth - q) / (q + c.eps) * 100.0
        # q_rec_exact_err = torch.abs(q_rec_exact - q) / (q + c.eps) * 100.0
        q_rec_rel_diff = (
            torch.abs(q_rec_exact - q_rec_smooth)
            / (q_rec_exact + c.eps)
            * 100.0
        )
        # q_rec_rel_diff = torch.abs(q_rec_exact - q_rec_smooth)

        plt.clf()
        set_plot(1)
        plt.plot(d.t, d.obs_data[idx].squeeze(), **get_opts(1, 1))
        # plt.plot(d.t, d.rdata[idx].squeeze(), **get_opts(1, 2))
        plt.legend(**c.sub.legend[0])

        set_plot(2)
        plt.plot(d.t, d.cdf[idx].squeeze(), **get_opts(2, 1))

        set_plot(3)
        plt.plot(d.p, d.q_exact[idx].squeeze(), **get_opts(3, 1))
        plt.ylim(d.t.min(), d.t.max())

        set_plot(4)
        plt.plot(d.p, d.q_deriv_exact[idx].squeeze(), **get_opts(4, 1))
        plt.plot(d.p, d.q_deriv_smooth[idx].squeeze(), **get_opts(4, 2))
        plt.legend(**c.sub.legend[3])

        set_plot(5)
        plt.plot(
            d.p,
            q_rec_exact,
            **get_opts(5, 1),
        )
        plt.plot(d.p, q_rec_smooth, **get_opts(5, 2))
        # plt.plot(d.p, q, **get_opts(5, 3))
        # plt.legend(**c.sub.legend[4])

        set_plot(6)
        plt.plot(d.p, q_rec_rel_diff, **get_opts(6, 1))
        # plt.plot(d.p, q_rec_exact_err, **get_opts(6, 1))
        # plt.plot(d.p, q_rec_smooth_err, **get_opts(6, 2))
        # plt.legend(**c.sub.legend[5])
        return {}

    fig, axes = plt.subplots(*c.sub.shape, **c.sub.kw)
    plt.subplots_adjust(**c.sub.adjust)
    iter = bool_slice(*d.obs_data.shape, **c.iter[0])
    frames = get_frames_bool(
        data=None,
        iter=iter,
        fig=fig,
        axes=axes,
        plotter=plotter,
    )
    save_frames(frames, **c.save.spline)


if __name__ == '__main__':
    main()


def ignore():
    nx, nf, ns, eta, eta_x = 100, 5, 5, 0.1, 0.0
    x = torch.linspace(0, 4 * np.pi, 100)
    xnew = x + eta_x * torch.rand(nx)
    xnew = xnew.sort().values
    freq = torch.linspace(1, 1.5, nf)
    shift = torch.linspace(1, 1.5, ns)
    ground_truth = torch.sin(
        x[None, None, :] * freq[None, :, None] + shift[:, None, None]
    )
    y = ground_truth + torch.randn(ns, nf, nx) * eta

    z = y.numpy()
    x_np = x.numpy()
    tck = [splrep(x_np, z[i, j, :], s=0) for i in range(ns) for j in range(nf)]
    tck_s = [
        splrep(x_np, z[i, j, :], s=2) for i in range(ns) for j in range(nf)
    ]

    splines = []
    spline_s = []
    for i in range(len(tck)):
        print(f'Processing spline {i}')
        splines.append(BSpline(*tck[i])(xnew.numpy()))
        spline_s.append(BSpline(*tck_s[i])(xnew.numpy()))

    def plotter(*, data, idx, fig, axes, spline_s, x, y, ground_truth):
        plt.clf()
        axes = plt.gca()
        axes.tick_params(axis='x', length=0)
        axes.tick_params(axis='y', length=0)
        axes.set_xticks([])
        axes.set_yticks([])
        plt.scatter(
            x,
            y[idx, :].squeeze(),
            label='Obs Data',
            marker='o',
            color='black',
            s=20,
        )
        plt.plot(
            xnew, data[idx[0]], label='Exact Int.', linestyle='-', color='blue'
        )
        plt.plot(
            xnew,
            spline_s[idx[0]],
            label='Approx. Int.',
            linestyle=':',
            color='red',
        )
        plt.plot(
            x,
            ground_truth[idx[0]].squeeze(),
            label='Truth',
            marker='x',
            color='green',
        )
        plt.legend(loc='upper left', fontsize=8, bbox_to_anchor=(1.0, 1.0))
        plt.title(f'idx={idx}')
        plt.xlim([x[0], x[-1]])
        # plt.ylim([-1 - eta, 1 + eta])
        return {
            'spline_s': spline_s,
            'x': x,
            'y': y,
            'ground_truth': ground_truth,
        }

    fig, axes = plt.subplots()
    axes.tick_params(axis='x', length=0)
    axes.tick_params(axis='y', length=0)
    axes.set_xticks([])
    axes.set_yticks([])
    fig.subplots_adjust(left=0.1, right=0.8, top=0.9, bottom=0.1)
    iter = bool_slice(len(splines))
    frames = get_frames_bool(
        data=splines,
        iter=iter,
        fig=fig,
        axes=axes,
        plotter=plotter,
        spline_s=spline_s,
        x=x,
        y=y.reshape(ns * nf, nx),
        ground_truth=ground_truth.reshape(ns * nf, nx),
    )
    save_frames(frames, path='tmp', duration=1000)
