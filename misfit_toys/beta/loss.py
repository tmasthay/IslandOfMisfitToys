from typing import Callable

import torch
from mh.core import DotDict
from returns.curry import curry
from torch import fft

from misfit_toys.beta.prob import cdf, disc_quantile, get_quantile_lambda, pdf
from misfit_toys.utils import all_detached_cpu


@curry
def w2(f, *, renorm, x, p, tol=1.0e-04, max_iters=20, eps=1.0e-04):
    fr = renorm(f, x)
    Q = get_quantile_lambda(
        fr, x, p=p, renorm=renorm, tol=tol, max_iters=max_iters
    )

    def helper(g, renorm_func=renorm, eps_dummy=eps, Q_dummy=Q):
        # print('PDF...', end='', flush=True)
        # tmp = pdf(g, x, renorm=renorm_func, dim=-1)
        # tmp = g / torch.trapz(g, x, dim=-1)
        tmp = renorm_func(g + eps_dummy, x)
        tmp = tmp / torch.trapz(tmp, x, dim=-1)
        CDF = cdf(tmp, x, dim=-1)
        T = Q_dummy(CDF, deriv=False) - x
        integrand = T**2 * tmp
        res = torch.trapz(integrand, x, dim=-1)
        return res.sum()

    # return helper, Q
    return helper


def mse(f):
    def helper(g):
        return torch.nn.functional.mse_loss(f, g)

    return helper


@curry
def w2_reg(f, *, renorm, x, p, scale, tol=1.0e-04, max_iters=20):
    fr = renorm(f, x)
    Q = get_quantile_lambda(
        fr, x, p=p, renorm=renorm, tol=tol, max_iters=max_iters
    )

    def helper(g, renorm_func=renorm):
        # print('PDF...', end='', flush=True)
        # tmp = pdf(g, x, renorm=renorm_func, dim=-1)
        # tmp = g / torch.trapz(g, x, dim=-1)
        tmp = renorm_func(g, x)
        tmp = tmp / torch.trapz(tmp, x, dim=-1)
        CDF = cdf(tmp, x, dim=-1)
        T = Q(CDF, deriv=False) - x
        integrand = T**2 * tmp
        res = torch.trapz(integrand, x, dim=-1)
        return res.sum() + scale * torch.diff(g, dim=-1).pow(2).sum()

    # return helper, Q
    return helper


@curry
def w2_trunc(
    f, *, renorm, x, p, tol=1.0e-04, max_iters=20, eps=0.0, p_cutoff=1.0e-03
):
    fr = renorm(f, x)
    Q = get_quantile_lambda(
        fr, x, p=p, renorm=renorm, tol=tol, max_iters=max_iters
    )
    Qeval = Q(p).squeeze()
    left_idx = 0
    right_idx = len(p) - 1
    while Qeval[left_idx] < p_cutoff:
        left_idx += 1
    while Qeval[right_idx] > 1 - p_cutoff:
        right_idx -= 1

    fr_left = fr[:left_idx]
    fr_right = fr[right_idx:]

    def helper(
        g,
        renorm_func=renorm,
        eps_dummy=eps,
        Q_dummy=Q,
        fr_left_dummy=fr_left,
        fr_right_dummy=fr_right,
    ):
        tmp = renorm_func(g + eps_dummy, x)
        tmp = tmp / torch.trapz(tmp, x, dim=-1)
        CDF = cdf(tmp, x, dim=-1)[left_idx:right_idx]
        T = Q_dummy(CDF, deriv=False) - x[left_idx:right_idx]

        integrand = T**2 * tmp[left_idx:right_idx]
        res = torch.trapz(integrand, x[left_idx:right_idx], dim=-1)
        left_err = (g[:left_idx] - fr_left_dummy).pow(2).sum()
        right_err = (g[right_idx:] - fr_right_dummy).pow(2).sum()
        mid_err = res.sum()
        return mid_err + left_err + right_err

    # return helper, Q
    return helper


@curry
def pdf_match(f, *, renorm, x):
    fr = renorm(f, x)

    def helper(g, *, renorm=renorm, x=x, fr=fr):
        gr = renorm(g, x)
        int_history = DotDict({'meta': {'x': x}, 'data': {'fr': fr, 'gr': gr}})
        return torch.nn.functional.mse_loss(fr, gr), int_history

    return helper


@curry
def cdf_match(
    f,
    *,
    renorm: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x: torch.Tensor,
):
    fr = renorm(f, x)
    fr_cdf = cdf(fr, x, dim=-1)
    # fr_cdf = torch.cumulative_trapezoid(fr, x, dim=-1)

    def helper(
        g, *, fr_dummy=fr, fr_cdf_dummy=fr_cdf, renorm_dummy=renorm, x_dummy=x
    ):
        gr = renorm_dummy(g, x_dummy)
        gr_cdf = cdf(gr, x, dim=-1)
        # gr_cdf = torch.cumulative_trapezoid(gr, x, dim=-1)
        xd = x_dummy.detach().cpu()
        int_history = all_detached_cpu(
            DotDict(
                {
                    'PDF': {
                        'x': xd,
                        'fr': fr_dummy,
                        'gr': gr,
                    },
                    'CDF': {
                        'x': xd,
                        'fr_cdf': fr_cdf_dummy,
                        'g_cdf': gr_cdf,
                    },
                },
            )
        )
        res = (
            torch.nn.functional.mse_loss(fr_cdf_dummy, gr_cdf)
            + (fr[0] - gr[0]) ** 2
        )
        return res, int_history
        # return torch.nn.functional.mse_loss(fr, gr), int_history

    return helper


@curry
def quantile_match(
    f,
    *,
    renorm: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    p: torch.Tensor,
):
    fr = renorm(f, x)
    fr_cdf = cdf(fr, x, dim=-1)
    # frq = disc_quantile(fr_cdf, x, p=p)
    frq = get_quantile_lambda(f, x=x, p=p, renorm=renorm)
    # fr_cdf = torch.cumulative_trapezoid(fr, x, dim=-1)
    disc_frq = frq(p).squeeze()

    def helper(
        g,
        *,
        f_dummy=f,
        fr_dummy=fr,
        fr_cdf_dummy=fr_cdf,
        renorm_dummy=renorm,
        x_dummy=x,
        frq_dummy=frq,
        p_dummy=p,
    ):
        gr = renorm_dummy(g, x_dummy)
        gr_cdf = cdf(gr, x, dim=-1)
        # grq = disc_quantile(gr_cdf, x, p=p_dummy)
        T = frq_dummy(gr_cdf, deriv=False).squeeze() - x_dummy

        left, right = 10, 10
        right = len(T) - right
        T = T[left:right]
        # gr_cdf = torch.cumulative_trapezoid(gr, x, dim=-1)
        integrand = T**2 * gr[left:right]
        res = integrand.sum()  # + (f_dummy[0] - g[0]) ** 2
        xd = x_dummy.detach().cpu()
        grq = disc_quantile(gr_cdf, x, p=p)
        int_history = all_detached_cpu(
            DotDict(
                {
                    'PDF': {
                        'x': xd,
                        'fr': fr_dummy,
                        'gr': gr,
                    },
                    'CDF': {
                        'x': xd,
                        'fr_cdf': fr_cdf_dummy,
                        'g_cdf': gr_cdf,
                    },
                    'Transport': {
                        'x': xd[left:right],
                        'T deviation': T,
                        'integrand': T**2 * gr[left:right],
                    },
                    'Quantiles': {
                        'x': p_dummy,
                        'frq': disc_frq,
                        'grq': grq,
                    },
                },
            )
        )
        # res = torch.nn.functional.mse_loss(FQ, GQ)
        # first_diff = (fr[0] - gr[0]) ** 2
        # return res + first_diff, int_history
        return res, int_history
        # return torch.nn.functional.mse_loss(fr, gr), int_history

    return helper


@curry
def sobolev(f, *, scale, x):
    fhat = fft.fft(f)
    N = f.shape[-1]
    freqs = fft.fftfreq(N, d=x[1] - x[0]).to(x.device)
    kernel = (1.0 + freqs**2) ** (scale)

    def helper(
        g,
        *,
        lcl_x=x,
        lcl_fhat=fhat,
        lcl_kernel=kernel,
        lcl_f=f,
    ):
        ghat = fft.fft(g)
        integrand = (ghat - lcl_fhat).abs() ** 2 * lcl_kernel

        int_history = all_detached_cpu(
            DotDict(
                {
                    'ref': {'x': lcl_x, 'obs_data': lcl_f, 'guess': g},
                    'freq_domain': {'x': freqs, 'kernel': lcl_kernel},
                    'diff_freq': {'x': freqs, 'integrand': integrand},
                }
            )
        )
        return torch.trapz(integrand, freqs), int_history

    return helper


def huber(f, *, delta):
    def helper(g):
        diff = f - g
        abs_diff = diff.abs()
        mask = abs_diff < delta
        return torch.where(
            mask, abs_diff**2, 2 * delta * abs_diff - delta**2
        ).sum()

    return helper


@curry
def l1(f):
    def helper(g):
        return (f - g).abs().sum()

    return helper
