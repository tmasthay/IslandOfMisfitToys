"""
Data generation module (will be deprecated soon).
"""

import os

import torch


def random_trace(path):
    """
    Returns a random trace from the given path.

    Args:
        path (str): The path to the file containing the traces.

    Returns:
        torch.Tensor: A random trace from the file.
    """
    path = path.replace('conda', os.environ['CONDA_PREFIX'])
    u = torch.load(path)
    u = u.reshape(-1, u.shape[-1])
    idx = torch.randint(0, u.shape[0], (1,)).item()
    while u[idx].norm() < 1e-3:
        idx = torch.randint(0, u.shape[0], (1,)).item()
    return u[idx]


def uniform(x):
    """
    Returns a tensor of ones with the same shape as the input tensor x.

    Args:
        x (torch.Tensor): The input tensor.

    Returns:
        torch.Tensor: A tensor of ones with the same shape as x.
    """
    return torch.ones_like(x)


def sine_wave(x):
    """
    Computes the sine of the input value.

    Args:
        x (torch.Tensor): The input value.

    Returns:
        torch.Tensor: The sine of the input value.
    """
    return torch.sin(x)


def gaussian(x):
    """
    Computes the Gaussian distribution for the given input.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Gaussian distribution tensor.

    """
    u = torch.exp(-((x - 0.5) ** 2) / (2 * 0.2**2))
    u = u / torch.trapz(u, x, dim=-1)
    return u


def gaussian_mixture(x, *, mu, sigma, weights):
    """
    Computes the Gaussian mixture distribution for the given input values.

    Args:
        x (torch.Tensor): Input values.
        mu (List[float]): List of means for each Gaussian component.
        sigma (List[float]): List of standard deviations for each Gaussian component.
        weights (List[float]): List of weights for each Gaussian component.

    Returns:
        torch.Tensor: The computed Gaussian mixture distribution.

    """
    u = torch.zeros_like(x)
    for m, s, w in zip(mu, sigma, weights):
        u += w * torch.exp(-((x - m) ** 2) / (2 * s**2))
    u = u / torch.trapz(u, x, dim=-1)
    return u
