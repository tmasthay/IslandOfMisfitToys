"""
This is centralized suite of renormalization functions for Wasserstein-2.
This is intended to make an easy way to configure with misfit_toys.examples.hydra.main.
"""

import torch
from mh.core import draise
from returns.curry import curry


def ensure_norm(f):
    """
    Ensures that the output of the given function `f` is normalized.

    Args:
        f (callable): The function to be wrapped.

    Returns:
        callable: The wrapped function that ensures the output is normalized.
    """

    @curry
    def helper(u, x, **kwargs):
        v = f(u, **kwargs)
        # draise(v.shape, x.shape)
        return v / torch.trapz(v, x, dim=-1).unsqueeze(-1)

    return helper


@ensure_norm
def identity(u):
    """
    Returns the input value `u` unchanged.

    Args:
        u: The input value.

    Returns:
        The input value `u`.
    """
    return u


@ensure_norm
def abs_renorm(u):
    """
    Compute the absolute value of a tensor and apply renormalization.

    Args:
        u (torch.Tensor): The input tensor.

    Returns:
        torch.Tensor: The renormalized tensor.

    """
    return torch.abs(u)


@ensure_norm
def square_renorm(u):
    """
    Squares the input value and applies renormalization.

    Args:
        u (float): The input value.

    Returns:
        float: The squared and renormalized value.
    """
    return u**2


@ensure_norm
def softplus(u, *, k=1.0):
    """
    Applies the softplus function element-wise to the input tensor.

    Args:
        u (torch.Tensor): The input tensor.
        k (float, optional): The scaling factor. Default is 1.0.

    Returns:
        torch.Tensor: The output tensor after applying the softplus function.
    """
    return torch.log(1 + torch.exp(k * u)) / k


@ensure_norm
def relu(u):
    """
    Applies the rectified linear unit (ReLU) activation function element-wise.

    Args:
        u (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Output tensor with ReLU applied element-wise.
    """
    return torch.nn.functional.relu(u)


@ensure_norm
def const(u, *, c):
    """
    Adds a constant value to the input.

    Args:
        u: The input value.
        c: The constant value to be added.

    Returns:
        The sum of the input value and the constant value.
    """
    return u + c
