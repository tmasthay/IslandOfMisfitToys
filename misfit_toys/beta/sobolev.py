"""
This module contains the Sobolev norm misfit function.
STATUS: Broken.
"""

import torch
from mh.core import draise
from returns.curry import curry


def riel_legacy(*, obs_data, t, alpha):
    """
    Computes the Riemann-Liouville functional for a given function.

    Args:
        obs_data (torch.Tensor): The observed data.
        t (torch.Tensor): The time values.
        alpha (float): The alpha value.

    Returns:
        function: A helper function that computes the Riel legacy function for a given input function.

    """
    kernel = t ** (alpha - 1)
    fint_obs_data = torch.cumulative_trapezoid(kernel * obs_data, t, dim=-1)

    def helper(f):
        fint_f = torch.cumulative_trapezoid(kernel * f, t, dim=-1)
        integrand = fint_f - fint_obs_data
        return torch.sum(integrand**2)

    return helper


def riel(*, obs_data, t, alpha):
    """
    Computes the Riemann-Liouville fractional integral of a 1D function.

    Args:
        obs_data (torch.Tensor): The observed data.
        t (torch.Tensor): The time values.
        alpha (float): The alpha value.

    Returns:
        function: A helper function that computes the Riel functional for a given function.

    """
    kernel = t ** (alpha - 1)
    fint_obs_data = torch.cumulative_trapezoid(kernel * obs_data, t, dim=-1)

    def helper(f):
        fint_f = torch.cumulative_trapezoid(kernel * f, t, dim=-1)
        integrand = (fint_f - fint_obs_data) ** 2
        return integrand.sum()

    return helper
