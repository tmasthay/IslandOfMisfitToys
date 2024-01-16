from abc import abstractmethod
import torch
import torch.nn as nn
from torchcubicspline import natural_cubic_spline_coeffs, NaturalCubicSpline

import numpy as np
from itertools import product


def unbatch_splines(coeffs):
    assert len(coeffs) == 5
    assert coeffs[1].shape[-1] == 1
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


def unbatch_spline_eval(splines, t):
    assert t.shape[:-1] == splines.shape
    t = t.unsqueeze(-1)
    res = torch.empty(t.shape)
    for idx in product(*map(range, t.shape[:-2])):
        # t_idx = (*idx, slice(None), slice(0, 1))
        res[idx] = splines[idx].evaluate(t[idx]).squeeze(-1)
    return res.squeeze(-1)


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
        )
        slices = [
            slice(None) if i != dim else slice(1, None)
            for i in range(len(u.shape))
        ]
        v[slices] = u
        return v
    return u


# Function to compute true_quantile for an arbitrary shape torch tensor along its last dimension
def true_quantile(
    pdf,
    x,
    p,
    *,
    dx=None,
):
    if len(pdf.shape) == 1:
        if dx is not None:
            cdf = cum_trap(pdf, dx=dx, dim=-1)
        else:
            input(dx)
            cdf = cum_trap(pdf, x, dim=-1)
        indices = torch.searchsorted(cdf, p)
        indices = torch.clamp(indices, 0, len(x) - 1)
        return x[indices]
    else:
        # Initialize an empty tensor to store the results
        result_shape = pdf.shape[:-1]
        results = torch.empty(result_shape + (len(p),), dtype=torch.float32)

        input(results.shape)
        # Loop through the dimensions
        for idx in product(*map(range, result_shape)):
            pdf_slice = pdf[idx]
            results[idx] = true_quantile(pdf_slice, x, p, dx=dx)
            # results[idx] = torch.stack([x_slice, cdf_slice], dim=0)
        input(results.shape)
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
):
    q = true_quantile(pdf, x, p, dx=dx)
    if q.shape[-1] != 1:
        q = q.unsqueeze(-1)
    coeffs = natural_cubic_spline_coeffs(p, q)
    splines = unbatch_splines(coeffs)
    return splines


def w2_builder(is_const):
    if is_const:

        def helper(f, t, *, quantiles):
            F = cum_trap(f, dx=t[1] - t[0], dim=-1, preserve_dims=True)
            off_diag = unbatch_spline_eval(quantiles, F)
            while len(t.shape) < len(off_diag.shape):
                t = t.unsqueeze(-1)
            input(t.shape)
            return torch.trapezoid(
                (t - off_diag) ** 2 * f, dx=t[1] - t[0], dim=-1
            )

    else:

        def helper(f, t, *, quantiles):
            F = cum_trap(f, x=t, dim=-1, preserve_dims=True)
            off_diag = unbatch_spline_eval(quantiles, F)
            return torch.trapezoid((t - off_diag) ** 2 * f, x=t, dim=-1)

    return helper


w2_const = w2_builder(True)
w2 = w2_builder(False)


# class W2Loss(nn.Module):
#     def __init__(self, t, R, quantiles):
#         super().__init__()
#         self.R = R
#         self.quantiles = quantiles
#         self.t = t

#     def forward(self, f, g):
#         # Apply the transformation R
#         f_tilde = self.R(f, dt=self.t[1] - self.t[0])
#         transport = self.q

#         # Sort the distributions for inverse CDF computation
#         sorted_f, _ = torch.sort(f_tilde, dim=1)
#         sorted_g, _ = torch.sort(g_tilde, dim=1)

#         # Compute the cumulative sums to approximate CDFs
#         cum_f = torch.cumsum(sorted_f, dim=1)
#         cum_g = torch.cumsum(sorted_g, dim=1)

#         # Compute the inverse CDFs
#         inv_cdf_f = torch.linspace(
#             0, 1, steps=f.shape[1], device=f.device
#         ).expand_as(cum_f)
#         inv_cdf_g = torch.linspace(
#             0, 1, steps=g.shape[1], device=g.device
#         ).expand_as(cum_g)

#         # Compute the W2 distance using the inverse CDFs
#         w2_distance = torch.sqrt(torch.sum((inv_cdf_f - inv_cdf_g) ** 2, dim=1))

#         return torch.mean(w2_distance)


# def str_to_renorm(key):
#     def abs_renorm(y):
#         return torch.abs(y) / torch.sum(torch.abs(y), dim=1, keepdim=True)

#     def square_renorm(y):
#         return y**2 / torch.sum(y**2, dim=1, keepdim=True)

#     options = {'abs': abs_renorm, 'square': square_renorm}
#     return options[key]


class W2LossAbstract(torch.nn.Module):
    def __init__(self, *, R, quantiles, t):
        super().__init__()
        self.R = R
        self.quantiles = quantiles
        self.t = t

    @abstractmethod
    def forward(self, f):
        pass


class W2LossConst(W2LossAbstract):
    def forward(self, f):
        return w2_const(f, self.t, quantiles=self.quantiles)


class W2Loss(W2LossAbstract):
    def forward(self, f):
        return w2(f, self.t, quantiles=self.quantiles)
