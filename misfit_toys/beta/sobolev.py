import torch
from returns.curry import curry


def riel_legacy(*, obs_data, t, alpha):
    kernel = t ** (alpha - 1)
    fint_obs_data = torch.cumulative_trapezoid(kernel * obs_data, t, dim=-1)

    def helper(f):
        fint_f = torch.cumulative_trapezoid(kernel * f, t, dim=-1)
        integrand = fint_f - fint_obs_data
        return torch.sum(integrand**2)

    return helper


def riel(*, obs_data, t, alpha):
    # kernel = t ** (alpha - 1)
    # fint_obs_data = torch.cumulative_trapezoid(kernel * obs_data, t, dim=-1)

    def helper(f):
        # fint_f = torch.cumulative_trapezoid(kernel * f, t, dim=-1)
        # integrand = fint_f - fint_obs_data
        # return torch.sum(integrand**2)
        return ((obs_data - f) ** 2).sum()

    return helper
