import numpy as np
from scipy.interpolate import splrep, BSpline
from torchcubicspline import NaturalCubicSpline, natural_cubic_spline_coeffs
import torch
import matplotlib.pyplot as plt
from masthay_helpers.typlotlib import get_frames_bool, save_frames
from misfit_toys.utils import bool_slice

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
tck_s = [splrep(x_np, z[i, j, :], s=2) for i in range(ns) for j in range(nf)]

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
    return {'spline_s': spline_s, 'x': x, 'y': y, 'ground_truth': ground_truth}


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
