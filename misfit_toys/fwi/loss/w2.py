from abc import abstractmethod
from itertools import product
from time import time
from typing import Callable

from matplotlib import pyplot as plt
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
    """
    Unbatches a set of spline coefficients.

    Args:
        coeffs (list): A list of spline coefficients.

    Returns:
        ndarray: An array of NaturalCubicSpline objects.

    Raises:
        ValueError: If the shape of coeffs[1] is not (..., 1).

    Example:
        coeffs = [
            np.array([1, 2, 3]),
            np.array([[4], [5], [6]]),
            np.array([[7], [8], [9]]),
            np.array([[10], [11], [12]]),
            np.array([[13], [14], [15]])
        ]
        unbatch_splines(coeffs)
    """
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
    """
    Evaluates a batch of splines at given parameter values.

    Args:
        splines (torch.Tensor): A tensor containing the batch of splines.
        t (torch.Tensor): A tensor containing the parameter values at which to evaluate the splines.
        deriv (bool, optional): If True, evaluates the derivative of the splines. Defaults to False.

    Returns:
        torch.Tensor: A tensor containing the evaluated values of the splines.

    Raises:
        ValueError: If the shape of t does not match the shape of splines.

    """
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
        if deriv:
            res[idx] = splines[idx].derivative(t[idx]).squeeze(-1)
        else:
            res[idx] = splines[idx].evaluate(t[idx]).squeeze(-1)
    return res.squeeze(-1)


def spline_func(t, y):
    """
    Creates a spline function based on the given data points.

    Args:
        t (array-like): The input values.
        y (array-like): The corresponding output values.

    Returns:
        callable: A function that can be used to evaluate the spline at any given point.

    """
    coeffs = natural_cubic_spline_coeffs(t, y)
    splines = unbatch_splines(coeffs)

    def helper(t, *, deriv=False):
        return unbatch_spline_eval(splines, t, deriv=deriv)

    return helper


def cum_trap(y, x=None, *, dx=None, dim=-1, preserve_dims=True):
    """
    Computes the cumulative trapezoidal integration of `y` along the specified dimension.

    Args:
        y (torch.Tensor): The input tensor.
        x (torch.Tensor, optional): The optional input tensor representing the spacing between adjacent elements of `y`.
        dx (float, optional): The optional spacing between adjacent elements of `y`. If provided, `x` should be None.
        dim (int, optional): The dimension along which to compute the cumulative trapezoidal integration. Default is -1.
        preserve_dims (bool, optional): Whether to preserve the dimensions of the input tensor. Default is True.

    Returns:
        torch.Tensor: The tensor containing the cumulative trapezoidal integration of `y` along the specified dimension.

    Raises:
        ValueError: If both `x` and `dx` are provided.

    """
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
    """
    Computes the cumulative distribution function (CDF) of the input tensor.

    Args:
        y (Tensor): The input tensor.
        x (Tensor, optional): The x-values corresponding to the y-values. If not provided, the x-values are assumed to be evenly spaced.
        dx (float, optional): The spacing between consecutive x-values. If provided, the x-values are assumed to be evenly spaced.
        dim (int, optional): The dimension along which to compute the CDF. Default is -1.

    Returns:
        Tensor: The computed CDF tensor.

    Raises:
        ValueError: If both x and dx are provided.

    """
    if dx is not None:
        u = cum_trap(y, dx=dx, dim=dim)
    else:
        u = cum_trap(y, x, dim=dim)
    return u / u[..., -1].unsqueeze(-1)


# Function to compute true_quantile for an arbitrary shape torch tensor along its last dimension
def true_quantile(
    pdf, x, p, *, dx=None, ltol=0.0, rtol=0.0, atol=1e-2, err_top=50
):
    """
    Computes the true quantile of a probability distribution function (PDF).

    Args:
        pdf (torch.Tensor): The probability distribution function.
        x (torch.Tensor): The values corresponding to the PDF.
        p (torch.Tensor): The quantile probabilities.
        dx (float, optional): The spacing between values in `x`. Defaults to None.
        ltol (float, optional): The left tolerance for the CDF. Defaults to 0.0.
        rtol (float, optional): The right tolerance for the CDF. Defaults to 0.0.
        atol (float, optional): The absolute tolerance for the CDF. Defaults to 1e-2.
        err_top (int, optional): The maximum number of elements to display in error messages. Defaults to 50.

    Returns:
        torch.Tensor: The quantile values corresponding to the given probabilities.

    Raises:
        ValueError: If the CDFs are not within the expected range.

    """
    if len(pdf.shape) == 1:
        if dx is not None:
            cdf = cum_trap(pdf, dx=dx, dim=-1)
        else:
            cdf = cum_trap(pdf, x, dim=-1)
        if not torch.allclose(
            cdf[..., 0], torch.zeros(1).to(cdf.device), atol=atol
        ) or not torch.allclose(
            cdf[..., -1], torch.ones(1).to(cdf.device), atol=atol
        ):
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
            # print(idx)
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

        return results


def true_quantile_choppy(
    pdf, x, p, *, dx=None, ltol=0.0, rtol=0.0, atol=1e-2, err_top=50
):
    """
    Computes the true quantile of a given probability distribution function (pdf).

    Args:
        pdf (torch.Tensor): The probability distribution function.
        x (torch.Tensor): The values corresponding to the pdf.
        p (torch.Tensor): The probabilities at which to compute the quantile.
        dx (float, optional): The spacing between values in `x`. Defaults to None.
        ltol (float, optional): The left tolerance for the cumulative distribution function (cdf). Defaults to 0.0.
        rtol (float, optional): The right tolerance for the cumulative distribution function (cdf). Defaults to 0.0.
        atol (float, optional): The absolute tolerance for the cdf. Defaults to 1e-2.
        err_top (int, optional): The maximum number of elements to display in error messages. Defaults to 50.

    Returns:
        torch.Tensor: The computed quantile values.

    Raises:
        ValueError: If the cumulative distribution function (cdf) is not within the expected range.

    """
    if len(pdf.shape) == 1:
        if dx is not None:
            cdf = cum_trap(pdf, dx=dx, dim=-1)
        else:
            cdf = cum_trap(pdf, x, dim=-1)
        if not torch.allclose(
            cdf[..., 0], torch.zeros(1).to(cdf.device), atol=atol
        ) or not torch.allclose(
            cdf[..., -1], torch.ones(1).to(cdf.device), atol=atol
        ):
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
        return res
    else:
        # Initialize an empty tensor to store the results
        result_shape = pdf.shape[:-1]
        results = torch.empty(
            result_shape + (p.shape[-1],), dtype=torch.float32
        )
        # Loop through the dimensions
        for idx in product(*map(range, result_shape)):
            results[idx] = true_quantile(pdf[idx], x, p, dx=dx)

        return results


def cts_quantile(
    pdf, x, p, *, dx=None, ltol=0.0, rtol=0.0, atol=1e-2, err_top=50
):
    """
    Computes the continuous quantile function using natural cubic splines.

    Args:
        pdf (Tensor): The probability density function.
        x (Tensor): The input values.
        p (Tensor): The probabilities.
        dx (Tensor, optional): The spacing between consecutive x values. Defaults to None.
        ltol (float, optional): The left tolerance for the quantile function. Defaults to 0.0.
        rtol (float, optional): The right tolerance for the quantile function. Defaults to 0.0.
        atol (float, optional): The absolute tolerance for the quantile function. Defaults to 1e-2.
        err_top (int, optional): The maximum number of iterations for the quantile function. Defaults to 50.

    Returns:
        Tensor: The natural cubic spline coefficients.

    Raises:
        ValueError: If p is not a 1D or 2D tensor.

    """
    q = true_quantile(
        pdf, x, p, dx=dx, ltol=ltol, rtol=rtol, atol=atol, err_top=err_top
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


def quantile(pdf, x, p, *, dx=None, ltol=0.0, rtol=0.0, atol=1e-2, err_top=50):
    """
    Computes the quantile function for a given probability distribution.

    Args:
        pdf (callable): The probability density function.
        x (array-like): The input values at which to evaluate the quantile function.
        p (array-like): The probabilities at which to compute the quantile values.
        dx (float, optional): The spacing between consecutive values in `x`. Defaults to None.
        ltol (float, optional): The left tolerance for the quantile computation. Defaults to 0.0.
        rtol (float, optional): The right tolerance for the quantile computation. Defaults to 0.0.
        atol (float, optional): The absolute tolerance for the quantile computation. Defaults to 1e-2.
        err_top (int, optional): The maximum number of error terms to use in the quantile computation. Defaults to 50.

    Returns:
        callable: A function that computes the quantile values for a given input value `t`.

    """
    splines = cts_quantile(
        pdf, x, p, dx=dx, ltol=ltol, rtol=rtol, atol=atol, err_top=err_top
    )

    def helper(t, *, deriv=False):
        return unbatch_spline_eval(splines, t, deriv=deriv)

    return helper


def w2_builder(is_const):
    """
    Builds a helper function for calculating the W2 loss.

    Args:
        is_const (bool): Indicates whether the input function is constant or not.

    Returns:
        helper (function): The helper function for calculating the W2 loss.

    Raises:
        None

    """
    if is_const:

        def helper(f, t, *, quantiles):
            """
            Calculates the W2 loss for a constant input function.

            Args:
                f (torch.Tensor): The input function.
                t (torch.Tensor): The time values.
                quantiles (torch.Tensor): The quantiles.

            Returns:
                torch.Tensor: The W2 loss.

            Raises:
                None

            """
            F = cum_trap(f, dx=t[1] - t[0], dim=-1, preserve_dims=True)
            off_diag = unbatch_spline_eval(quantiles, F)
            dt = t[1].item() - t[0].item()
            while len(t.shape) < len(off_diag.shape):
                t = t.unsqueeze(0)
            return torch.trapezoid((t - off_diag) ** 2 * f, dx=dt, dim=-1)

    else:

        def helper(f, t, *, quantiles):
            """
            Calculates the W2 loss for a non-constant input function.

            Args:
                f (torch.Tensor): The input function.
                t (torch.Tensor): The time values.
                quantiles (torch.Tensor): The quantiles.

            Returns:
                torch.Tensor: The W2 loss.

            Raises:
                None

            """
            F = cum_trap(f, x=t, dim=-1, preserve_dims=True)
            off_diag = unbatch_spline_eval(quantiles, F)
            return torch.trapezoid((t - off_diag) ** 2 * f, x=t, dim=-1)

    return helper


w2_const = w2_builder(True)
w2 = w2_builder(False)


def simple_quantile(cdf_spline, *, p, tol=1e-4, max_iter=100, start, end):
    """
    Calculates the quantile values for given probabilities using a binary search algorithm.

    Args:
        cdf_spline (callable): A function that returns the cumulative distribution function (CDF) value at a given point.
        p (torch.Tensor): The probabilities for which quantile values need to be calculated.
        tol (float, optional): The tolerance value for convergence. Defaults to 1e-4.
        max_iter (int, optional): The maximum number of iterations for the binary search algorithm. Defaults to 100.
        start (float): The starting point of the search space.
        end (float): The ending point of the search space.

    Returns:
        torch.Tensor: The quantile values corresponding to the given probabilities.

    Raises:
        ValueError: If the quantile calculation does not converge for any of the probabilities.
    """
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
    """
    W2Loss calculates the W2 Wasserstein loss between the given traces and the observed data.

    Args:
        t (torch.Tensor): The time values.
        p (torch.Tensor): The probability values.
        obs_data (torch.Tensor): The observed data.
        renorm (callable): A function to renormalize the data.
        gen_deriv (callable): A function to generate derivatives.
        down (int, optional): The downsampling factor. Defaults to 1.

    Attributes:
        obs_data (torch.Tensor): The renormalized observed data.
        renorm (callable): The renormalization function.
        q_raw (torch.Tensor): The true quantile values.
        p (torch.Tensor): The probability values.
        t (torch.Tensor): The time values.
        q (callable): The spline function.
        qd (callable): The derivative function.

    """

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
        if gen_deriv is None:
            self.qd = lambda a,b: None
        else:
            self.qd = gen_deriv(q=self.q, p=self.p)

    def forward(self, traces):
        """
        Calculates the W2 Wasserstein loss between the given traces and the observed data.

        Args:
            traces (torch.Tensor): The input traces.

        Returns:
            torch.Tensor: The calculated loss.

        """
        pdf = self.renorm(traces)
        cdf = cum_trap(pdf, self.t)
        transport = self.q(cdf)
        diff = self.t - transport
        loss = torch.trapz(diff**2 * pdf, self.t, dim=-1)
        return loss
    
class W2LossScalar(W2Loss):
    """
    W2LossScalar calculates the W2 Wasserstein loss between the given traces and the observed data.

    Args:
        t (torch.Tensor): The time values.
        p (torch.Tensor): The probability values.
        obs_data (torch.Tensor): The observed data.
        renorm (callable): A function to renormalize the data.
        gen_deriv (callable): A function to generate derivatives.
        down (int, optional): The downsampling factor. Defaults to 1.

    Attributes:
        obs_data (torch.Tensor): The renormalized observed data.
        renorm (callable): The renormalization function.
        q_raw (torch.Tensor): The true quantile values.
        p (torch.Tensor): The probability values.
        t (torch.Tensor): The time values.
        q (callable): The spline function.
        qd (callable): The derivative function.

    """

    def __init__(self, *, t, p, obs_data, renorm, down=1):
        super().__init__(t=t, p=p, obs_data=obs_data, renorm=renorm, gen_deriv=None, down=down)
    
    def forward(self, traces):
        """
        Calculates the W2 Wasserstein loss between the given traces and the observed data.

        Args:
            traces (torch.Tensor): The input traces.
        """
        return super().forward(traces).sum()
    
if __name__ == "__main__":
    t = torch.linspace(-10,10,1000)
    p = torch.linspace(0,1,10000)
    
    N = 25
    mu = torch.linspace(1, 2, N)
    sig = torch.linspace(0.1, 1.0, N)
    
    u = (t[None, None, :] - mu[:, None, None])**2 / (2 * sig[None, :, None]**2)
    
    formal_pdf = torch.exp(-u)
    
    def simple_renorm(y):
        z = torch.abs(y)
        return z / torch.trapz(z, dx=t[1]-t[0], dim=-1)[..., None]
    
    loser = W2Loss(t=t, p=p, obs_data=formal_pdf, renorm=simple_renorm, down=1, gen_deriv=None)
    
    mid_mu = mu[len(mu) // 2]
    mid_sig = sig[len(sig) // 2]
    ref_pdf = formal_pdf[len(mu) // 2, len(sig) // 2, :]
    
    loss = loser(ref_pdf)
    
    analytic_solution = (mu[:, None] - mid_mu)**2 + (sig[None, :]-mid_sig)**2
    diff = torch.abs(loss - analytic_solution)
    
    plt.imshow(diff, cmap='seismic', aspect='auto')
    plt.colorbar()
    plt.savefig('mine.jpg')

