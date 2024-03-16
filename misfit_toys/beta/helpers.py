import os
import sys
from matplotlib import pyplot as plt
import torch
from dataclasses import dataclass
import torch.nn.functional as F
from torchcubicspline import NaturalCubicSpline, natural_cubic_spline_coeffs
import numpy as np
from scipy.stats import norm
from mh.core import draise

from misfit_toys.utils import bool_slice
from mh.typlotlib import get_frames_bool, save_frames
from mh.core import DotDict, torch_stats

torch.set_printoptions(
    precision=4, sci_mode=False, callback=torch_stats(report='all')
)


@dataclass
class Defaults:
    cdf_tol: float = 1.0e-03


def unbatch_splines(x, y):
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


def unbatch_splines_lambda(x, y):
    splines = unbatch_splines(x, y)
    splines_flattened = splines.flatten()

    def helper(z, *, sf=splines_flattened, shape=splines.shape):
        # z2 = z.expand(*shape, -1)
        res = torch.stack([e.evaluate(z) for e in sf], dim=0)
        return res.view(*shape, -1)

    return helper


def pdf(u, x, *, renorm):
    v = u if renorm is None else renorm(u, x)
    err = torch.abs(torch.trapz(v, x, dim=-1) - 1.0)
    if torch.where(err > Defaults.cdf_tol, 1, 0).any():
        draise(f"PDF is not normalized: err = {err}")
    return v


def cdf(pdf, x, *, dim=-1):
    u = torch.cumulative_trapezoid(pdf, x, dim=dim)

    pad = [0] * (2 * u.dim())
    pad[2 * dim + 1] = 1
    pad.reverse()

    return F.pad(u, pad, value=0.0)


def disc_quantile(cdfs, x, *, p):
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


# This is a performance bottleneck
#     May consider using cython or numba
#     -> numba doesn't work with pytorch
#
# Another approach that might be faster since it crank on GPU...
#     Simply upsample cdfs until the max(torch.diff(cdfs) < tol) is satisfied
#     Then use torch.searchsorted...this would guarantee that the quantile is
#         no worse than tol.
def cts_quantile_legacy(cdfs, x, *, p, tol=1.0e-04, max_iters=20):
    cdf_splines = unbatch_splines(x, cdfs).flatten()
    q = torch.empty(cdf_splines.shape[0], p.shape[0]).to(p.device)
    for i in range(cdf_splines.shape[0]):
        print(f'{i}/{cdf_splines.shape[0]}')
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
                    f"Quantile not found for p = {pp}, left={left}, right={right}, curr={curr}, start={start}, end={end}, mid={mid}"
                )


def cts_quantile(cdfs, x, *, p, tol=1.0e-04, max_iters=20):
    cdf_splines = unbatch_splines(x, cdfs).flatten()
    q = torch.empty(cdf_splines.shape[0], p.shape[0]).to(p.device)
    for i in range(cdf_splines.shape[0]):
        print(f'{i}/{cdf_splines.shape[0]}')
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
                    f"Quantile not found for p = {pp}, left={left}, right={right}, curr={curr}, start={start}, end={end}, mid={mid}"
                )

    q = q.view(*cdfs.shape[:-1], p.shape[0])
    Q = unbatch_splines_lambda(p, q)
    # Q = unbatch_splines(p, q)
    return Q


def get_quantile_lambda(u, x, *, p, renorm, tol=1.0e-04, max_iters=20):
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
