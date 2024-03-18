import os
import sys
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from mh.core import DotDict, draise, torch_stats
from mh.typlotlib import get_frames_bool, save_frames
from numpy.typing import NDArray
from scipy.stats import norm
from torchcubicspline import NaturalCubicSpline, natural_cubic_spline_coeffs

from misfit_toys.utils import bool_slice

# torch.set_printoptions(
#     precision=4, sci_mode=False, callback=torch_stats(report='all')
# )


@dataclass
class Defaults:
    cdf_tol: float = 1.0e-03


def unbatch_splines(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    N = max(1, np.prod(y.shape[:-1]))
    if x.dim() == 1:
        x = x.expand(N, -1)
    u = np.empty(N, dtype=object)
    coeffs = np.empty(N, dtype=object)
    z = y.reshape(-1, y.shape[-1], 1)
    for i in range(N):
        coeffs[i] = natural_cubic_spline_coeffs(x[i], z[i])
        u[i] = NaturalCubicSpline(coeffs[i])

    shape = (1,) if y.dim() == 1 else y.shape[:-1]
    u = u.reshape(*shape)
    return u


def unbatch_splines_lambda(
    x: torch.Tensor, y: torch.Tensor
) -> Callable[[torch.Tensor, NaturalCubicSpline, tuple], torch.Tensor]:
    splines = unbatch_splines(x, y)
    splines_flattened = splines.flatten()

    def helper(z, *, sf=splines_flattened, shape=splines.shape):
        # z2 = z.expand(*shape, -1)
        res = torch.stack([e.evaluate(z) for e in sf], dim=0)
        return res.view(*shape, -1)

    return helper


def pdf(
    u: torch.Tensor,
    x: torch.Tensor,
    *,
    renorm: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    v = u if renorm is None else renorm(u, x)
    err = torch.abs(torch.trapz(v, x, dim=-1) - 1.0)
    if torch.where(err > Defaults.cdf_tol, 1, 0).any():
        draise(f"PDF is not normalized: err = {err}")
    return v


def cdf(pdf: torch.Tensor, x: torch.Tensor, *, dim=-1) -> torch.Tensor:
    u = torch.cumulative_trapezoid(pdf, x, dim=dim)

    pad = [0] * (2 * u.dim())
    pad[2 * dim + 1] = 1
    pad.reverse()

    return F.pad(u, pad, value=0.0)


def disc_quantile(
    cdfs: torch.Tensor, x: torch.Tensor, *, p: torch.Tensor
) -> torch.Tensor:
    if x.dim() == 1:
        x = x.expand(cdfs.shape[:-1] + (-1,))
    indices = torch.searchsorted(
        cdfs, p.expand(*cdfs.shape[:-1], -1), right=False
    )
    right_indices = torch.clamp(indices, 0, cdfs.shape[-1] - 1)
    left_vals, right_vals = x.gather(-1, indices), x.gather(-1, right_indices)
    left_cdfs, right_cdfs = cdfs.gather(-1, indices), cdfs.gather(
        -1, right_indices
    )
    denom = right_cdfs - left_cdfs
    denom = torch.where(denom < 1.0e-8, 1.0, denom)
    alpha = (p - left_cdfs) / denom
    return left_vals + alpha * (right_vals - left_vals)


def cts_quantile(
    cdfs: torch.Tensor,
    x: torch.Tensor,
    *,
    p: torch.Tensor,
    tol: float = 1.0e-04,
    max_iters: int = 20,
):
    cdf_splines = unbatch_splines(x, cdfs).flatten()
    q = torch.empty(cdf_splines.shape[0], p.shape[0]).to(p.device)
    for i in range(cdf_splines.shape[0]):
        start, end = x[0], x[-1]
        for j, pp in enumerate(p):
            for guesses in range(max_iters):
                mid = (start + end) / 2
                left = cdf_splines[i].evaluate(start)
                right = cdf_splines[i].evaluate(end)
                curr = cdf_splines[i].evaluate(mid)
                if pp < left:
                    start = (x[0] + mid) / 2
                elif pp > right:
                    end = (x[-1] + mid) / 2
                elif abs(pp - curr) < tol:
                    q[i, j] = mid
                    break
                elif pp < curr:
                    end = mid
                else:
                    start = mid
            if guesses >= max_iters - 1:
                raise ValueError(
                    f"Quantile not found for p = {pp}, left={left},"
                    f" right={right}, curr={curr}, start={start}, end={end},"
                    f" mid={mid}"
                )

    q = q.view(*cdfs.shape[:-1], p.shape[0])
    Q = unbatch_splines_lambda(p, q)
    # Q = unbatch_splines(p, q)
    return Q


def get_quantile_lambda(
    u: torch.Tensor,
    x: torch.Tensor,
    *,
    p: torch.Tensor,
    renorm: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    tol=1.0e-04,
    max_iters=20,
):
    PDF = pdf(u, x, renorm=renorm)
    CDF = cdf(PDF, x)
    if p.dim() != 1:
        raise NotImplementedError(
            f"Only 1D p is supported for now, got p.shape = {p.shape}"
        )
    Q = cts_quantile(CDF, x, p=p, tol=tol, max_iters=max_iters)
    return Q


def combos(*args, flat=True):
    u = torch.stack(torch.meshgrid(*args), dim=-1)
    return u.view(-1, len(args)) if flat else u


# def main1():
#     x = torch.linspace(-5, 5, 25)
#     eps = 0.01
#     p = torch.linspace(eps, 1.0 - eps, 100)
#     u = torch.exp(-(x**2))

#     # renorm = lambda u: u / torch.trapz(u, x, dim=-1)
#     def renorm(u):
#         return u / torch.trapz(u, x, dim=-1)

#     PDF = pdf(u, x, renorm=renorm)
#     CDF = cdf(PDF, x)
#     Q = disc_quantile(CDF, x, p=p)

#     x_up = torch.linspace(-5, 5, 500)
#     CDF_up = unbatch_splines_lambda(x, CDF)(x_up).squeeze()
#     Q_updirect = disc_quantile(CDF_up, x_up, p=p)
#     Qspline = cts_quantile(CDF, x, p=p)
#     Q_up = Qspline(p).squeeze()

#     import os

#     import matplotlib.pyplot as plt

#     shape = (3, 1)
#     plt.subplots(*shape, figsize=(10, 10))
#     plt.subplot(*shape, 1)
#     plt.plot(x, PDF)
#     plt.title('PDF')

#     plt.subplot(*shape, 2)
#     plt.plot(x, CDF, label='original')
#     plt.plot(x_up, CDF_up, label='interpolated')
#     plt.legend(framealpha=0.2)
#     plt.title('CDF')

#     plt.subplot(*shape, 3)
#     plt.plot(p, Q, label='Q')
#     plt.plot(p, Q_updirect, label='Q from upsampled CDF')
#     plt.plot(p, Q_up, 'ro', label='Q from full spline', markersize=1)
#     plt.legend()
#     plt.title('Quantile')
#     out_file = os.path.join(__file__, '..', 'prob.jpg')
#     out_file = os.path.abspath(out_file)
#     plt.savefig(out_file)

#     print(f'\nSee {out_file}\n')


# def main2():
#     x = torch.linspace(-5, 5, 25)
#     eps = 0.01
#     p = torch.linspace(eps, 1.0 - eps, 100)
#     mu, sig = 0, 1
#     u = torch.exp(-((x - mu) ** 2 / (2 * sig**2)))

#     # renorm = lambda u: u / torch.trapz(u, x, dim=-1)
#     def renorm(u):
#         return u / torch.trapz(u, x, dim=-1)

#     Q = get_quantile_lambda(u, x, p=p, renorm=renorm)
#     Qref = norm.ppf(p, loc=mu, scale=sig)
#     input(Q(p).shape)

#     shape = (2, 1)
#     plt.subplots(*shape, figsize=(10, 10))
#     plt.subplot(*shape, 1)
#     plt.plot(x, u)
#     plt.title('Raw Data')

#     plt.subplot(*shape, 2)
#     plt.plot(p, Q(p).squeeze(), label='Q')
#     plt.plot(p, Qref, label='Qref')
#     plt.legend(framealpha=0.5)
#     plt.title('Quantile')

#     out_file = os.path.join(__file__, '..', 'prob2.jpg')
#     out_file = os.path.abspath(out_file)
#     plt.savefig(out_file)

#     print(f'\nSee {out_file}\n')


# def main3():
#     x = torch.linspace(-5, 5, 25)
#     eps = 0.01
#     p = torch.linspace(eps, 1.0 - eps, 100)
#     mu = torch.linspace(-2, 2, 3)
#     sig = torch.linspace(0.5, 1.5, 7)
#     u = torch.exp(
#         -((x[None, None, :] - mu[:, None, None]) ** 2)
#         / (2 * sig[None, :, None] ** 2)
#     )

#     def renorm(v):
#         return v / torch.trapz(v, x, dim=-1).unsqueeze(-1)

#     Q = get_quantile_lambda(u, x, p=p, renorm=renorm)
#     Qcomp = Q(p).squeeze().detach()
#     Qref = [norm.ppf(p, loc=m, scale=s) for m, s in combos(mu, sig, flat=True)]
#     Qref = np.array(Qref).reshape(*Qcomp.shape)

#     Qflat = Qcomp.view(-1, Qcomp.shape[-1])
#     input([e.data_ptr() for e in Qflat])

#     def plotter(*, data, idx, fig, axes, shape):
#         plt.clf()
#         plt.suptitle(
#             r'$\mu={:.2f}, \sigma={:.2f}$'.format(
#                 data.mu[idx[0]], data.sig[idx[1]]
#             )
#         )
#         plt.subplot(*shape, 1)
#         plt.plot(data.x, data.raw[idx])
#         plt.ylim(data.raw.min(), data.raw.max())
#         plt.title('Raw Data')

#         plt.subplot(*shape, 2)
#         plt.plot(data.p, data.Q[idx], label=r'$Q$')
#         plt.plot(
#             data.p, data.Qref[idx], 'ro', markersize=1, label=r'Analytic $Q$'
#         )
#         plt.legend(framealpha=0.3)
#         plt.title('Quantile')
#         plt.ylim(data.x.min(), data.x.max())
#         plt.tight_layout()
#         return {'shape': shape}

#     shape = (2, 1)
#     fig, axes = plt.subplots(*shape, figsize=(10, 10))
#     iter = bool_slice(*Qcomp.shape, none_dims=[-1])
#     frames = get_frames_bool(
#         iter=iter,
#         plotter=plotter,
#         data=DotDict(dict(x=x, p=p, raw=u, Q=Qcomp, Qref=Qref, mu=mu, sig=sig)),
#         shape=shape,
#         fig=fig,
#         axes=axes,
#     )
#     save_frames(frames, path='prob3.gif', duration=1000)

#     out_file = os.path.join(__file__, '..', 'prob3.gif')
#     out_file = os.path.abspath(out_file)
#     print(f'\nSee {out_file}\n')


# def main4():
#     x = torch.linspace(-5, 5, 25)
#     eps = 0.01
#     p = torch.linspace(eps, 1.0 - eps, 100)
#     mu = torch.linspace(-2, 2, 3)
#     sig = torch.linspace(0.5, 1.5, 7)
#     u = torch.exp(
#         -((x[None, None, :] - mu[:, None, None]) ** 2)
#         / (2 * sig[None, :, None] ** 2)
#     )

#     pdf_ref = u * (1 / (sig[None, :, None] * np.sqrt(2 * np.pi)))
#     cdf_ref = 0.5 * (
#         1.0
#         + torch.erf(
#             (x[None, None, :] - mu[:, None, None])
#             / (sig[None, :, None] * np.sqrt(2))
#         )
#     )
#     Qref = mu[:, None, None] + sig[None, :, None] * np.sqrt(2) * torch.erfinv(
#         2 * p - 1
#     )

#     def renorm(v, y):
#         return v / torch.trapz(v, y, dim=-1).unsqueeze(-1)

#     PDF = pdf(u, x, renorm=renorm)
#     CDF = cdf(PDF, x)
#     # Q = get_quantile_lambda(u, x, p=p, renorm=renorm)
#     Q = cts_quantile(CDF, x, p=p)
#     # Q = Q.reshape(int(np.prod(Q.shape)))
#     # Qcomp = torch.Tensor(np.array([e.evaluate(p) for e in Q])).squeeze()
#     # Qcomp = Qcomp.view(mu.numel(), sig.numel(), -1)
#     # input(Qcomp.shape)
#     Qcomp = Q(p).squeeze().detach()
#     # Qcomp = disc_quantile(CDF, x, p=p)
#     # Qref = [norm.ppf(p, loc=m, scale=s) for m, s in combos(mu, sig, flat=True)]
#     # Qref = np.array(Qref).reshape(*Qcomp.shape)

#     def plotter(*, data, idx, fig, axes, shape):
#         plt.clf()
#         plt.suptitle(
#             r'$\mu={:.2f}, \sigma={:.2f}$'.format(
#                 data.mu[idx[0]], data.sig[idx[1]]
#             )
#         )
#         plt.subplot(*shape, 1)
#         plt.plot(data.x, data.raw[idx], label='Computed PDF')
#         plt.plot(
#             data.x, data.pdf_ref[idx], 'ro', label='Analytic PDF', markersize=1
#         )
#         plt.ylim(data.raw.min(), data.raw.max())
#         plt.title('PDF')

#         plt.subplot(*shape, 2)
#         plt.plot(data.x, data.cdf[idx], label='Computed CDF')
#         plt.plot(
#             data.x, data.cdf_ref[idx], 'ro', label='Analytic CDF', markersize=1
#         )
#         plt.title('CDF')

#         plt.subplot(*shape, 3)
#         plt.plot(data.p, data.Q[idx], label=r'$Q$')
#         plt.plot(
#             data.p, data.Qref[idx], 'ro', markersize=1, label=r'Analytic $Q$'
#         )
#         plt.legend(framealpha=0.3)
#         plt.title('Quantile')
#         plt.ylim(data.x.min(), data.x.max())
#         plt.tight_layout()
#         return {'shape': shape}

#     d = DotDict(
#         dict(
#             x=x,
#             p=p,
#             raw=PDF,
#             cdf=CDF,
#             Q=Qcomp,
#             Qref=Qref,
#             mu=mu,
#             sig=sig,
#             cdf_ref=cdf_ref,
#             pdf_ref=pdf_ref,
#         )
#     )
#     shape = (3, 1)
#     fig, axes = plt.subplots(*shape, figsize=(10, 10))
#     iter = bool_slice(*Qcomp.shape, none_dims=[-1])
#     frames = get_frames_bool(
#         iter=iter,
#         plotter=plotter,
#         data=d,
#         shape=shape,
#         fig=fig,
#         axes=axes,
#     )
#     save_frames(frames, path='prob4.gif', duration=1000)

#     out_file = os.path.join(__file__, '..', 'prob4.gif')
#     out_file = os.path.abspath(out_file)
#     print(f'\nSee {out_file}\n')

#     return d


# def main5():
#     d = main4()
#     return d


# if __name__ == "__main__":
#     mode = 4 if len(sys.argv) == 1 else int(sys.argv[1])
#     if mode == 1:
#         main1()
#     elif mode == 2:
#         main2()
#     elif mode == 3:
#         main3()
#     elif mode == 4:
#         main4()
#     else:
#         raise ValueError(f"Invalid mode: {mode}")
