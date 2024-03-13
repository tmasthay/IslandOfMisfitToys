from abc import abstractmethod
from itertools import product
from time import time
from typing import Callable

import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
from mh.core import DotDict
from returns.curry import curry
from scipy.ndimage import median_filter, uniform_filter
from torchcubicspline import NaturalCubicSpline, natural_cubic_spline_coeffs

from misfit_toys.utils import bool_slice, tensor_summary


def unbatch_splines(coeffs):
    assert len(coeffs) == 5
    if coeffs[1].shape[-1] != 1:
        raise ValueError(
            'Expected coeffs[1].shape[-1] to be 1, got ---'
            f' coeffs[1].shape = {coeffs[1].shape}'
        )
    target_shape = coeffs[1].shape[:-2]
    res = np.empty(target_shape, dtype=object)
    for idx in product(*map(range, target_shape)):
        curr = (
            coeffs[0],
            coeffs[1][idx],
            coeffs[2][idx],
            coeffs[3][idx],
            coeffs[4][idx],
        )
        res[idx] = NaturalCubicSpline(curr)
    return res


def unbatch_spline_eval(splines, t, *, deriv=False):
    if len(t.shape) == 1:
        t = t.expand(*splines.shape, -1)
    if t.shape[:-1] != splines.shape:
        raise ValueError(
            'Expected t.shape[:-1] to be equal to splines.shape, got ---'
            f' t.shape = {t.shape}, splines.shape = {splines.shape}'
        )
    t = t.unsqueeze(-1)
    res = torch.empty(t.shape).to(t.device)
    for idx in product(*map(range, t.shape[:-2])):
        # t_idx = (*idx, slice(None), slice(0, 1))
        if deriv:
            res[idx] = splines[idx].derivative(t[idx]).squeeze(-1)
        else:
            res[idx] = splines[idx].evaluate(t[idx]).squeeze(-1)
    return res.squeeze(-1)


def spline_func(t, y):
    coeffs = natural_cubic_spline_coeffs(t, y)
    splines = unbatch_splines(coeffs)

    def helper(t, *, deriv=False):
        return unbatch_spline_eval(splines, t, deriv=deriv)

    return helper


def cum_trap(y, x=None, *, dx=None, dim=-1, preserve_dims=True):
    if dx is not None:
        u = torch.cumulative_trapezoid(y, dx=dx, dim=dim)
    else:
        u = torch.cumulative_trapezoid(y, x, dim=dim)
    if preserve_dims:
        if dim < 0:
            dim = len(u.shape) + dim
        v = torch.zeros(
            [e if i != dim else e + 1 for i, e in enumerate(u.shape)]
        ).to(u.device)
        slices = [
            slice(None) if i != dim else slice(1, None)
            for i in range(len(u.shape))
        ]
        v[slices] = u
        return v
    return u


def get_cdf(y, x=None, *, dx=None, dim=-1):
    if dx is not None:
        u = cum_trap(y, dx=dx, dim=dim)
    else:
        u = cum_trap(y, x, dim=dim)
    return u / u[..., -1].unsqueeze(-1)


# Function to compute true_quantile for an arbitrary shape torch tensor along its last dimension
def true_quantile(
    pdf,
    x,
    p,
    *,
    dx=None,
    ltol=0.0,
    rtol=0.0,
    atol=1e-2,
    err_top=50,
):
    if len(pdf.shape) == 1:
        if dx is not None:
            cdf = cum_trap(pdf, dx=dx, dim=-1)
            # cdf_verify = torch.trapz(pdf, dx=dx, dim=-1)
        else:
            cdf = cum_trap(pdf, x, dim=-1)
            # cdf_verify = torch.trapezoid(pdf, x, dim=-1)
        if not torch.allclose(
            cdf[..., 0], torch.zeros(1).to(cdf.device), atol=atol
        ) or not torch.allclose(
            cdf[..., -1], torch.ones(1).to(cdf.device), atol=atol
        ):
            # flattened = cdf.reshape(-1)
            # left_disc = torch.topk(flattened, err_top, largest=False)[0]
            # right_disc = torch.topk(flattened, err_top, largest=True)[0]
            # pdf_mins = torch.topk(pdf.reshape(-1), err_top, largest=False)[0]
            # pdf_maxs = torch.topk(pdf.reshape(-1), err_top, largest=True)[0]
            raise ValueError(
                'CDFs should theoretically be in [0.0, 1.0] and in practice be'
                f' in [{atol}, {1.0 - atol}], observed info below\nCDF\n\n'
                f' {tensor_summary(cdf, err_top)}\nPDF\n\n{tensor_summary(pdf, err_top)}\n'
            )

        indices = torch.searchsorted(cdf, p)
        indices = torch.clamp(indices, 0, len(x) - 1)
        res = torch.tensor([x[i] for i in indices])

        return res
    else:
        # Initialize an empty tensor to store the results
        result_shape = pdf.shape[:-1]
        results = torch.empty(
            result_shape + (p.shape[-1],), dtype=torch.float32
        )
        # Loop through the dimensions
        for idx in product(*map(range, result_shape)):
            # results[idx] = true_quantile(pdf[idx], x[idx], p[idx], dx=dx)
            print(idx)
            results[idx] = true_quantile(
                pdf[idx],
                x,
                p,
                dx=dx,
                atol=atol,
                err_top=err_top,
                ltol=ltol,
                rtol=rtol,
            )
            # results[idx] = torch.stack([x_slice, cdf_slice], dim=0)
        # num_dims = len(results.shape)
        # permutation = (
        #     [num_dims - 2] + list(range(num_dims - 2)) + [num_dims - 1]
        # )

        # return results.permute(*permutation)
        return results


def true_quantile_choppy(
    pdf,
    x,
    p,
    *,
    dx=None,
    ltol=0.0,
    rtol=0.0,
    atol=1e-2,
    err_top=50,
):
    if len(pdf.shape) == 1:
        if dx is not None:
            cdf = cum_trap(pdf, dx=dx, dim=-1)
            # cdf_verify = torch.trapz(pdf, dx=dx, dim=-1)
        else:
            cdf = cum_trap(pdf, x, dim=-1)
            # cdf_verify = torch.trapezoid(pdf, x, dim=-1)
        if not torch.allclose(
            cdf[..., 0], torch.zeros(1).to(cdf.device), atol=atol
        ) or not torch.allclose(
            cdf[..., -1], torch.ones(1).to(cdf.device), atol=atol
        ):
            # flattened = cdf.reshape(-1)
            # left_disc = torch.topk(flattened, err_top, largest=False)[0]
            # right_disc = torch.topk(flattened, err_top, largest=True)[0]
            # pdf_mins = torch.topk(pdf.reshape(-1), err_top, largest=False)[0]
            # pdf_maxs = torch.topk(pdf.reshape(-1), err_top, largest=True)[0]
            raise ValueError(
                'CDFs should theoretically be in [0.0, 1.0] and in practice be'
                f' in [{atol}, {1.0 - atol}], observed info below\nCDF\n\n'
                f' {tensor_summary(cdf, err_top)}\nPDF\n\n{tensor_summary(pdf, err_top)}\n'
            )

        left_cutoff_idx = torch.where(cdf < ltol)[0]
        right_cutoff_idx = torch.where(cdf > 1 - rtol)[0]

        left_cutoff_idx = list(left_cutoff_idx) or 0
        right_cutoff_idx = list(right_cutoff_idx) or -1

        indices = torch.searchsorted(cdf, p)
        indices = torch.clamp(indices, 0, len(x) - 1)
        res = x[indices]
        # res[:left_cutoff_idx] = x[0]
        # res[right_cutoff_idx:] = x[-1]
        return res
    else:
        # Initialize an empty tensor to store the results
        result_shape = pdf.shape[:-1]
        results = torch.empty(
            result_shape + (p.shape[-1],), dtype=torch.float32
        )
        # Loop through the dimensions
        for idx in product(*map(range, result_shape)):
            # results[idx] = true_quantile(pdf[idx], x[idx], p[idx], dx=dx)
            results[idx] = true_quantile(pdf[idx], x, p, dx=dx)
            # results[idx] = torch.stack([x_slice, cdf_slice], dim=0)
        # num_dims = len(results.shape)
        # permutation = (
        #     [num_dims - 2] + list(range(num_dims - 2)) + [num_dims - 1]
        # )

        # return results.permute(*permutation)
        return results


def cts_quantile(
    pdf,
    x,
    p,
    *,
    dx=None,
    ltol=0.0,
    rtol=0.0,
    atol=1e-2,
    err_top=50,
):
    q = true_quantile(
        pdf,
        x,
        p,
        dx=dx,
        ltol=ltol,
        rtol=rtol,
        atol=atol,
        err_top=err_top,
    )
    if q.shape[-1] != 1:
        q = q.unsqueeze(-1)
    if len(p.shape) > 2:
        raise ValueError(
            f'Expected p to be a 1D or 2D tensor, got --- p.shape = {p.shape}'
        )
    elif len(p.shape) == 2:
        runner = [
            natural_cubic_spline_coeffs(p_slice, q_slice)
            for p_slice, q_slice in zip(p, q)
        ]
        splines = np.stack([unbatch_splines(e) for e in runner])
    else:
        coeffs = natural_cubic_spline_coeffs(p, q)
        splines = unbatch_splines(coeffs)
    return splines


def quantile(
    pdf,
    x,
    p,
    *,
    dx=None,
    ltol=0.0,
    rtol=0.0,
    atol=1e-2,
    err_top=50,
):
    splines = cts_quantile(
        pdf,
        x,
        p,
        dx=dx,
        ltol=ltol,
        rtol=rtol,
        atol=atol,
        err_top=err_top,
    )

    def helper(t, *, deriv=False):
        return unbatch_spline_eval(splines, t, deriv=deriv)

    return helper


def w2_builder(is_const):
    if is_const:

        def helper(f, t, *, quantiles):
            F = cum_trap(f, dx=t[1] - t[0], dim=-1, preserve_dims=True)
            off_diag = unbatch_spline_eval(quantiles, F)
            dt = t[1].item() - t[0].item()
            while len(t.shape) < len(off_diag.shape):
                t = t.unsqueeze(0)
            return torch.trapezoid((t - off_diag) ** 2 * f, dx=dt, dim=-1)

    else:

        def helper(f, t, *, quantiles):
            F = cum_trap(f, x=t, dim=-1, preserve_dims=True)
            off_diag = unbatch_spline_eval(quantiles, F)
            return torch.trapezoid((t - off_diag) ** 2 * f, x=t, dim=-1)

    return helper


w2_const = w2_builder(True)
w2 = w2_builder(False)


def simple_quantile(cdf_spline, *, p, tol=1e-4, max_iter=100, start, end):
    q = torch.zeros(p.shape).to(p.device)
    for i, prob in enumerate(p):
        left, right = start, end
        for iter in range(max_iter):
            mid = (left + right) / 2
            cdf_val = cdf_spline(mid)

            # Check if current mid value meets the probability with tolerance
            if abs(cdf_val - prob) < tol:
                q[i] = mid
                break

            # Adjust search space
            if cdf_val < prob:
                left = mid
                if cdf_spline(right) < prob:
                    right = (right + end) / 2
            else:
                right = mid
        if iter == max_iter - 1:
            raise ValueError(
                f"Quantile calculation did not converge for probability {prob}"
            )

    return q


class W2Loss(torch.nn.Module):
    def __init__(self, *, t, p, obs_data, renorm, gen_deriv, down=1):
        super().__init__()
        self.obs_data = renorm(obs_data)
        self.renorm = renorm
        self.q_raw = true_quantile(
            self.obs_data, t, p, rtol=0.0, ltol=0.0, err_top=10
        ).to(self.obs_data.device)
        self.p = p
        self.t = t
        self.q = spline_func(
            self.p[::down], self.q_raw[..., ::down].unsqueeze(-1)
        )
        self.qd = gen_deriv(q=self.q, p=self.p)

    def forward(self, traces):
        pdf = self.renorm(traces)
        cdf = cum_trap(pdf, self.t)
        transport = self.q(cdf)
        diff = self.t - transport
        loss = torch.trapz(diff**2 * pdf, self.t, dim=-1)
        return loss
