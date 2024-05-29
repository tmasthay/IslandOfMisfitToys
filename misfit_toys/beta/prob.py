"""
TODO: Deprecate this after verifying that it is not used.
"""

import os
import pickle
import sys
from dataclasses import dataclass
from itertools import product
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
    print_freq = N // 10
    for i in range(N):
        if i % print_freq == 0:
            print(f'{i=} / {N}', flush=True)
        coeffs[i] = natural_cubic_spline_coeffs(x[i], z[i])
        u[i] = NaturalCubicSpline(coeffs[i])

    shape = (1,) if y.dim() == 1 else y.shape[:-1]
    u = u.reshape(*shape)
    # debug.mark -> returns with right shape (16, 100) for marmousi
    return u


def unbatch_splines_lambda(
    x: torch.Tensor, y: torch.Tensor
) -> Callable[[torch.Tensor, bool], torch.Tensor]:
    splines = unbatch_splines(x, y)
    splines_flattened = splines.flatten()
    # draise(splines.shape, splines_flattened.shape)

    def helper(z, *, sf=splines_flattened, shape=splines.shape, deriv=False):
        # draise(z.shape)
        z = z.view(*splines_flattened.shape, -1)
        assert z.shape[0] == len(sf)
        # draise(z.shape, shape)
        if not deriv:
            res = torch.stack(
                [e.evaluate(z[i]) for i, e in enumerate(sf)], dim=0
            )
        else:
            res = torch.stack(
                [e.derivative(z[i]) for i, e in enumerate(sf)], dim=0
            )
        res = res.view(*shape, -1)
        # draise(res.shape)
        return res

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
        print('PDF is not normalized: err =', err, flush=True)
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
    if len(cdfs.shape) == 1:
        indices = torch.searchsorted(cdfs, p)
        indices = torch.clamp(indices, 0, len(x) - 1)
        res = torch.tensor([x[i] for i in indices])
        return res
    else:
        result_shape = cdfs.shape[:-1]
        results = torch.empty(
            result_shape + (p.shape[-1],), dtype=torch.float32
        )
        for idx in product(*map(range, result_shape)):
            results[idx] = disc_quantile(cdfs[idx], x, p=p)

        return results


def cts_quantile2(
    cdfs: torch.Tensor,
    x: torch.Tensor,
    *,
    p: torch.Tensor,
    tol: float = 1.0e-04,
    max_iters: int = 20,
):
    # temporary debugging
    # q_path = os.path.expanduser('~/important_data/q.pt')
    # if os.path.exists(q_path):
    #     q =˚ ˚torch.load(q_path)
    #     # draise(q.shape)
    #     q = q.view(*cdfs.shape[:-1], p.shape[0]).to(p.device)
    #     Q = unbatch_splines_lambda(p, q)
    #     print('Returning cached quantile', flush=True)
    #     return Q

    cdf_splines = unbatch_splines(x, cdfs).flatten()
    q = torch.empty(cdf_splines.shape[0], p.shape[0]).to(p.device)

    # print_freq = cdf_splines.shape[0] // 10
    print_freq = 1
    for i in range(cdf_splines.shape[0]):
        if i % print_freq == 0:
            print(f'cts_quantile: {i=} / {cdf_splines.shape[0]}', flush=True)
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

    # # pickle it for caching
    # with open('/tmp/cts_quantile.pkl', 'wb') as f:
    #     pickle.dump(Q, f)
    # np.save(f'/tmp/cts_quantile_{CURR_IDX}.npy', q.cpu().numpy())
    return Q


def cts_quantile(
    cdfs: torch.Tensor,
    x: torch.Tensor,
    *,
    p: torch.Tensor,
    tol: float = 1.0e-04,
    max_iters: int = 20,
):
    idx = torch.searchsorted(cdfs, p.expand(*cdfs.shape[:-1], -1))
    idx = torch.clamp(idx, 0, len(x) - 1)
    q = x[idx]
    Q = unbatch_splines_lambda(p, q)
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
