"""
Centralized loss functions for use in misfit_toys.examples.hydra.main.
This module also contains decorators to wrap around other loss functions.
For example, see tik_reg, which can add Tikhonov regularization to any loss function.
"""

import inspect
from functools import wraps
from typing import Callable

import torch
from mh.core import DotDict, draise
from returns.curry import curry
from torch import fft
from torchcubicspline import NaturalCubicSpline as NCS
from torchcubicspline import natural_cubic_spline_coeffs as ncs

from misfit_toys.beta.prob import cdf, disc_quantile, get_quantile_lambda, pdf
from misfit_toys.utils import all_detached_cpu


def linear_combo(*, losses, weights=None):
    """
    Computes a linear combination of multiple loss functions.

    Args:
        losses (list): A list of loss functions.
        weights (list, optional): A list of weights for each loss function. If not provided, equal weights are used.

    Returns:
        function: A function that takes an input `x` and returns the linear combination of the loss functions.

    Raises:
        AssertionError: If the number of losses is not equal to the number of weights.

    Example:
        >>> loss1 = lambda x: x**2
        >>> loss2 = lambda x: abs(x)
        >>> combined_loss = linear_combo(losses=[loss1, loss2], weights=[0.3, 0.7])
        >>> combined_loss(2)
        1.4
    """
    if weights is None:
        weights = torch.ones(len(losses))
    else:
        weights = torch.tensor(weights)
    assert len(losses) == len(weights)
    weights = weights / weights.sum()
    return lambda x: sum(w * f(x) for w, f in zip(weights, losses))


def scurry(**dec_kwargs):
    """
    A decorator factory that allows passing keyword arguments to the decorated function.

    Args:
        dec_kwargs: Keyword arguments to be passed to the decorated function.

    Returns:
        A decorator that adds the specified keyword arguments to the decorated function.

    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            all_kwargs = {**dec_kwargs, **kwargs}

            return func(*args, **all_kwargs)

        return wrapper

    return decorator


def lin_decrease(*, _min=1.0e-16, _max=1.0, max_calls):
    """
    Returns a helper function that implements linear decrease.

    The helper function calculates a value between `_min` and `_max` based on the number of calls made to it.
    The value decreases linearly from `_max` to `_min` over `max_calls` number of calls.

    Args:
        _min (float, optional): The minimum value. Defaults to 1.0e-16.
        _max (float, optional): The maximum value. Defaults to 1.0.
        max_calls (int): The maximum number of calls after which the value reaches `_min`.

    Returns:
        function: A helper function that calculates the linearly decreasing value.

    """
    num_calls = 0

    def helper():
        nonlocal num_calls
        num_calls += 1
        return _min + (_max - _min) * max(0.0, (1 - num_calls / max_calls))

    return helper


@curry
def tik_reg(
    f, *, model_params, base_loss, weights, penalty=None, reg_sched=None
):
    """Applies Tikhonov regularization to a loss function.

    Args:
        f: The loss function to be regularized.
        model_params: The parameters of the model.
        base_loss: The base loss function.
        weights: The weights for the misfit term and regularization term.
        penalty: The penalty factor for the regularization term (optional).
        reg_sched: The regularization schedule (optional).

    Returns:
        A helper function that applies Tikhonov regularization to the loss function.
    """
    if weights is None:
        weights = torch.ones(2)
    else:
        weights = torch.tensor(weights)

    misfit = base_loss(f)

    def helper(g):
        misfit_term = misfit(g)

        num_deriv = torch.diff(model_params(), dim=-1)

        val = num_deriv**2
        if penalty is not None:
            val = penalty * val
        reg = (num_deriv**2).mean()
        if reg_sched is not None:
            reg = reg_sched() * reg
        return weights[0] * misfit_term + weights[1] * reg

    return helper


@curry
def w2(f, *, renorm, x, p, tol=1.0e-04, max_iters=20, eps=1.0e-04):
    """
    Calculates the W2 Wasserstein distance between the input function `f` and a target distribution.

    Args:
        f: The input function.
        renorm: A function used to renormalize the input function.
        x: The input values.
        p: The quantile value.
        tol: The tolerance value for convergence (default: 1.0e-04).
        max_iters: The maximum number of iterations for convergence (default: 20).
        eps: A small value added to the input function to avoid division by zero (default: 1.0e-04).

    Returns:
        A function that calculates the W2 Wasserstein distance between the input function `f` and a target distribution.
    """
    fr = renorm(f, x)
    Q = get_quantile_lambda(
        fr, x, p=p, renorm=renorm, tol=tol, max_iters=max_iters
    )

    def helper(g):
        """
        Helper function that calculates the W2 Wasserstein distance between the input function `f` and `g`.

        Args:
            g: The target function.

        Returns:
            The W2 Wasserstein distance between the input function `f` and `g`.
        """
        # print('PDF...', end='', flush=True)
        # tmp = pdf(g, x, renorm=renorm_func, dim=-1)
        # tmp = g / torch.trapz(g, x, dim=-1)
        tmp = renorm(g + eps, x)
        tmp = tmp / torch.trapz(tmp, x, dim=-1)
        CDF = cdf(tmp, x, dim=-1)
        T = Q(CDF, deriv=False) - x
        integrand = T**2 * tmp
        res = torch.trapz(integrand, x, dim=-1)
        return res.sum()

    # return helper, Q
    return helper


def mse(f):
    """
    Calculates the mean squared error (MSE) loss between two tensors.

    Args:
        f (torch.Tensor): The first tensor.

    Returns:
        function: A helper function that takes in another tensor and calculates the MSE loss.

    """

    def helper(g):
        return torch.nn.functional.mse_loss(f, g)

    return helper


@curry
def w2_reg(f, *, renorm, x, p, scale, tol=1.0e-04, max_iters=20):
    """
    Computes the Wasserstein-2 regularization loss.

    Args:
        f (callable): The function to be regularized.
        renorm (callable): The renormalization function.
        x (torch.Tensor): The input tensor.
        p (float): The quantile value.
        scale (float): The scaling factor for the regularization term.
        tol (float, optional): The tolerance value for convergence. Defaults to 1.0e-04.
        max_iters (int, optional): The maximum number of iterations. Defaults to 20.

    Returns:
        callable: The helper function that computes the regularization loss.
    """
    fr = renorm(f, x)
    Q = get_quantile_lambda(
        fr, x, p=p, renorm=renorm, tol=tol, max_iters=max_iters
    )

    def helper(g, renorm_func=renorm):
        """
        Helper function that computes the regularization loss.

        Args:
            g (torch.Tensor): The input tensor.
            renorm_func (callable, optional): The renormalization function. Defaults to renorm.

        Returns:
            torch.Tensor: The regularization loss.
        """
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
    """
    Calculates the truncated Wasserstein-2 distance between two probability distributions.

    Args:
        f (callable): The function representing the probability distribution.
        renorm (callable): The function used to renormalize the probability distribution.
        x (torch.Tensor): The input tensor.
        p (torch.Tensor): The target probability distribution.
        tol (float, optional): The tolerance for convergence. Defaults to 1.0e-04.
        max_iters (int, optional): The maximum number of iterations. Defaults to 20.
        eps (float, optional): The epsilon value. Defaults to 0.0.
        p_cutoff (float, optional): The cutoff value for p. Defaults to 1.0e-03.

    Returns:
        callable: The helper function used for calculating the loss.
    """
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

    def helper(g):
        tmp = renorm(g + eps, x)
        tmp = tmp / torch.trapz(tmp, x, dim=-1)
        CDF = cdf(tmp, x, dim=-1)[left_idx:right_idx]
        T = Q(CDF, deriv=False) - x[left_idx:right_idx]

        integrand = T**2 * tmp[left_idx:right_idx]
        res = torch.trapz(integrand, x[left_idx:right_idx], dim=-1)
        left_err = (g[:left_idx] - fr_left).pow(2).sum()
        right_err = (g[right_idx:] - fr_right).pow(2).sum()
        mid_err = res.sum()
        return mid_err + left_err + right_err

    return helper


@curry
def pdf_match(f, *, renorm, x):
    """
    Returns a helper function that calculates the mean squared error (MSE) loss between two probability density functions (PDFs).

    Args:
        f: The first PDF function.
        renorm: A function used to renormalize the PDFs.
        x: The input value for the PDFs.

    Returns:
        A helper function that takes in a second PDF function and returns the MSE loss between the renormalized PDFs of f and g, as well as an intermediate history object.

    """
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
    """Compute the CDF match loss.

    Args:
        f: The input function.
        renorm: A function that normalizes the input function.
        x: The input tensor.

    Returns:
        A helper function that computes the CDF match loss and returns the loss value and intermediate history.
    """
    fr = renorm(f, x)
    fr_cdf = cdf(fr, x, dim=-1)

    def helper(
        g, *, fr_dummy=fr, fr_cdf_dummy=fr_cdf, renorm_dummy=renorm, x_dummy=x
    ):
        """Helper function that computes the CDF match loss.

        Args:
            g: The input function.
            fr_dummy: A dummy variable for fr.
            fr_cdf_dummy: A dummy variable for fr_cdf.
            renorm_dummy: A dummy variable for renorm.
            x_dummy: A dummy variable for x.

        Returns:
            The CDF match loss value and intermediate history.
        """
        gr = renorm_dummy(g, x_dummy)
        gr_cdf = cdf(gr, x, dim=-1)
        xd = x_dummy.detach().cpu()
        int_history = all_detached_cpu(
            DotDict(
                {
                    'PDF': {'x': xd, 'fr': fr_dummy, 'gr': gr},
                    'CDF': {'x': xd, 'fr_cdf': fr_cdf_dummy, 'g_cdf': gr_cdf},
                }
            )
        )
        res = (
            torch.nn.functional.mse_loss(fr_cdf_dummy, gr_cdf)
            + (fr[0] - gr[0]) ** 2
        )
        return res, int_history

    return helper


@curry
def quantile_match(
    f,
    *,
    renorm: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    p: torch.Tensor,
):
    """Quantile matching loss function.

    Args:
        f: The input function.
        renorm: A function that performs renormalization.
        x: The input tensor.
        p: The quantile tensor.

    Returns:
        A helper function that calculates the loss and returns the result along with the integration history.
    """
    fr = renorm(f, x)
    fr_cdf = cdf(fr, x, dim=-1)
    frq = get_quantile_lambda(f, x=x, p=p, renorm=renorm)
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
        """Helper function that calculates the loss and integration history.

        Args:
            g: The input tensor.
            f_dummy: A dummy variable for f.
            fr_dummy: A dummy variable for fr.
            fr_cdf_dummy: A dummy variable for fr_cdf.
            renorm_dummy: A dummy variable for renorm.
            x_dummy: A dummy variable for x.
            frq_dummy: A dummy variable for frq.
            p_dummy: A dummy variable for p.

        Returns:
            The loss value and the integration history.
        """
        gr = renorm_dummy(g, x_dummy)
        gr_cdf = cdf(gr, x, dim=-1)
        T = frq_dummy(gr_cdf, deriv=False).squeeze() - x_dummy

        left, right = 10, 10
        right = len(T) - right
        T = T[left:right]
        integrand = T**2 * gr[left:right]
        res = integrand.sum()
        xd = x_dummy.detach().cpu()
        grq = disc_quantile(gr_cdf, x, p=p)
        int_history = all_detached_cpu(
            DotDict(
                {
                    'PDF': {'x': xd, 'fr': fr_dummy, 'gr': gr},
                    'CDF': {'x': xd, 'fr_cdf': fr_cdf_dummy, 'g_cdf': gr_cdf},
                    'Transport': {
                        'x': xd[left:right],
                        'T deviation': T,
                        'integrand': T**2 * gr[left:right],
                    },
                    'Quantiles': {'x': p_dummy, 'frq': disc_frq, 'grq': grq},
                }
            )
        )
        return res, int_history

    return helper


@curry
def sobolev(f, *, scale, x):
    """
    Calculates the Sobolev loss between two functions.

    Args:
        f (torch.Tensor): The target function.
        scale (float): The scale parameter for the Sobolev kernel.
        x (torch.Tensor): The input values.

    Returns:
        tuple: A tuple containing the Sobolev loss and the integration history.

    """
    fhat = fft.fft(f)
    N = f.shape[-1]
    freqs = fft.fftfreq(N, d=x[1] - x[0]).to(x.device)
    kernel = (1.0 + freqs**2) ** (scale)

    def helper(g, *, lcl_x=x, lcl_fhat=fhat, lcl_kernel=kernel, lcl_f=f):
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
    """
    Computes the Huber loss between the input `f` and a target value `g`.

    Args:
        f (torch.Tensor): The input tensor.
        delta (float): The threshold value for the Huber loss.

    Returns:
        torch.Tensor: The computed Huber loss.

    """

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
    """
    Calculates the L1 loss between two tensors.

    Args:
        f (Tensor): The first tensor.

    Returns:
        Callable: A function that takes in another tensor and returns the L1 loss between the two tensors.
    """

    def helper(g):
        return (f - g).abs().mean()

    return helper


def l1_double(f, g):
    """
    Calculates the mean absolute difference between two tensors.

    Args:
        f (Tensor): The first tensor.
        g (Tensor): The second tensor.

    Returns:
        Tensor: The mean absolute difference between f and g.
    """
    return (f - g).abs().mean()


@curry
def transform_loss(f, *, loss, transform):
    """
    Applies a transformation function to the input functions and calculates the loss between the transformed functions.

    Args:
        f: The input function.
        loss: The loss function used to calculate the loss between the transformed functions.
        transform: The transformation function applied to the input functions.

    Returns:
        A helper function that takes another function as input, applies the same transformation, and calculates the loss between the transformed functions.

    """
    F = transform(f)

    def helper(g):
        G = transform(g)
        return loss(F, G)

    return helper


def softplus(*, scale=1.0, t):
    """
    Compute the softplus function.

    Args:
        scale (float, optional): Scaling factor for the input. Defaults to 1.0.
        t (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: The cumulative distribution function (CDF) of the softplus function.

    """

    def helper(f):
        u = torch.log(1 + torch.exp(scale * f)) / scale
        u = u / torch.trapz(u, t, dim=-1).unsqueeze(-1)
        CDF = torch.cumulative_trapezoid(u, t, dim=-1)
        return CDF

    return helper


def identity(f):
    """
    Returns the input function `f` unchanged.

    Args:
        f: The input function.

    Returns:
        The input function `f`.
    """
    return f


def inverse(f, x):
    """
    Computes the inverse of a function using natural cubic splines.

    Args:
        f: The function to compute the inverse of.
        x: The input value.

    Returns:
        The inverse of the function evaluated at the given input value.
    """
    coeffs = ncs(f, x.unsqueeze(-1))
    spline = NCS(coeffs)
    return spline
