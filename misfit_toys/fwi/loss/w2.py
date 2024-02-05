from abc import abstractmethod
from itertools import product

import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
from masthay_helpers.global_helpers import DotDict
from scipy.ndimage import median_filter, uniform_filter
from torchcubicspline import NaturalCubicSpline, natural_cubic_spline_coeffs

from misfit_toys.utils import tensor_summary


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

        last_low_idx = (cdf <= ltol).nonzero(as_tuple=True)[0][-1] + 1
        fst_hi_idx = (cdf >= 1 - rtol).nonzero(as_tuple=True)[0][0] - 1

        # input((cdf <= ltol).nonzero(as_tuple=True)[0][-1])

        # last_low_idx = 0
        # fst_hi_idx = len(x) - 2
        indices = torch.clamp(
            indices, last_low_idx, min(fst_hi_idx, len(x) - 2)
        )

        # integral = torch.floor(indices)
        # remainder = indices - integral
        # right_indices = (integral + 1).long()
        # integral = integral.long()
        # res = (1 - remainder) * x[integral] + remainder * x[right_indices]

        res = x[indices.long()]

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


class W2LossFunctionLegacy(autograd.Function):
    @staticmethod
    def forward(ctx, f, t, renorm, quantiles):
        # Unpack non-tensor arguments from args

        # Perform the forward computation
        t_expand = t.expand(*quantiles.shape, -1)
        f_tilde = renorm(f)
        F = cum_trap(f_tilde, dx=t[1] - t[0]).to(f.device)
        off_diag = unbatch_spline_eval(quantiles, F)
        diff = (t_expand - off_diag) ** 2
        integrated = torch.trapezoid(diff * f_tilde, dx=t[1] - t[0], dim=-1)

        # Save variables for backward pass
        ctx.save_for_backward(f, f_tilde, diff, integrated)
        ctx.renorm = renorm

        return integrated.sum()

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve saved variables
        # f, f_tilde, diff, integrated = ctx.saved_tensors
        # renorm = ctx.renorm

        # # Compute custom gradient
        # # grad_input = ... (implement custom gradient computation)

        # return grad_input
        return ctx + grad_output


class W2LossLegacy(nn.Module):
    def __init__(self, *, t, renorm, obs_data, p):
        super().__init__()
        self.obs_data = obs_data
        self.device = obs_data.device
        self.t = t.to(self.device)
        self.renorm = renorm
        self.renorm_obs_data = renorm(obs_data).to(self.device)
        self.quantiles = cts_quantile(self.renorm_obs_data, t, p)

    def forward(self, f):
        # Call the custom autograd function
        return W2LossFunction.apply(
            f, self.obs_data, self.t, self.renorm, self.quantiles
        )


def eval_w2(quantiles):
    def helper(*, pdf, t, cdf=None, return_all=True):
        t_expand = t.expand(*quantiles.shape, -1)
        if cdf is None:
            cdf = cum_trap(pdf, dx=t[1] - t[0], dim=-1)
        transport_map = unbatch_spline_eval(quantiles, cdf)
        diff = t_expand - transport_map
        res = torch.trapezoid(diff**2 * pdf, dx=t[1] - t[0], dim=-1)
        if return_all:
            return DotDict(
                {
                    'res': res,
                    'diff': diff,
                    'transport': transport_map,
                    'cdf': cdf,
                }
            )
        else:
            return res

    return helper


def eval_w2_grad(q, qd):
    def helper(*, pdf, cdf, transport, t):
        t_expand = t.expand(*q.shape, -1)
        diff = t_expand - transport
        integrand = -2 * diff * pdf * unbatch_spline_eval(qd, cdf)
        integral = cum_trap(integrand, dx=t[1] - t[0], dim=-1)
        reverse_integral = integral[..., -1].unsqueeze(-1) - integral
        res = diff**2 + reverse_integral
        return res

    return helper


def wass2(q, qd):
    return DotDict({'eval': eval_w2(q), 'grad': eval_w2_grad(q, qd)})


def quantile_deriv(pdfs, x, p, *, filter_func=None, deriv_filter_func=None):
    def iden(x):
        return x

    filter_func = iden if filter_func is None else filter_func
    deriv_filter_func = iden if deriv_filter_func is None else deriv_filter_func

    filtered_pdfs = filter_func(pdfs)
    filtered_pdfs = filtered_pdfs / torch.trapezoid(
        filtered_pdfs, dx=x[1] - x[0], dim=-1
    ).unsqueeze(-1)
    q = cts_quantile(filtered_pdfs, x, p)
    filtered_deriv = deriv_filter_func(unbatch_spline_eval(q, p, deriv=True))
    if filtered_deriv.shape[-1] != 1:
        filtered_deriv = filtered_deriv.unsqueeze(-1)
    # raise ValueError(filtered_deriv.norm())
    coeffs = natural_cubic_spline_coeffs(p, filtered_deriv)
    q_deriv = unbatch_splines(coeffs)
    return q, q_deriv


class W2LossFunction(autograd.Function):
    @staticmethod
    def forward(ctx, obs_data, t, renorm, q, qd):
        pdf = renorm(obs_data)
        cdf = get_cdf(pdf, x=t, dim=-1)
        transport = q(cdf)
        diff = t.expand(transport.shape) - transport
        res = torch.trapezoid(diff**2 * pdf, dx=t[1] - t[0], dim=-1).sum()
        ctx.save_for_backward(t, pdf, cdf, transport, diff)
        ctx.qd = qd
        return res

    @staticmethod
    def backward(ctx, grad_output):
        t, pdf, cdf, transport, diff = ctx.saved_tensors
        qd = ctx.qd
        diff = t.expand(transport.shape) - transport
        integrand = -2 * diff * pdf * qd(cdf)
        integral = cum_trap(integrand, dx=t[1] - t[0], dim=-1)
        reverse_integral = integral[..., -1].unsqueeze(-1) - integral
        res = diff**2 + reverse_integral
        return res * grad_output, None, None, None, None


class W2Loss(torch.nn.Module):
    def __init__(
        self, *, t, p, renorm, obs_data, smoother=None, ltol=1.0e-2, rtol=1.0e-2
    ):
        super().__init__()
        self.t = t
        self.renorm = renorm
        self.obs_data = obs_data
        self.obs_data_renorm = renorm(obs_data)
        self.q = quantile(self.obs_data_renorm, t, p, ltol=ltol, rtol=rtol)
        deriv = self.q(deriv=True)
        if smoother is not None:
            deriv = smoother(deriv)
        self.qd = spline_func(p, deriv)

    def forward(self, traces):
        return W2LossFunction.apply(traces, self.t, self.renorm, self.q)
