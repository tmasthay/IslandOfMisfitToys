from abc import ABC, abstractmethod
from typing import Callable

import torch
from mh.core import DotDict

from misfit_toys.beta.loss import l1_double, lin_decrease
from misfit_toys.beta.loss import softplus as softplus_loss
from misfit_toys.beta.loss import tik_reg, transform_loss
from misfit_toys.beta.renorm import softplus as softplus_renorm


def working_w1(
    obs_data, *, model_params, scale, weights, reg_min, t, max_calls
):
    return tik_reg(
        f=obs_data,
        model_params=model_params,
        base_loss=transform_loss(
            loss=l1_double, transform=softplus_loss(scale=scale, t=t)
        ),
        weights=weights,
        reg_sched=lin_decrease(max_calls=max_calls, _min=reg_min),
    )


def riel_transform(*, scale, t, eps):
    def helper(f):
        kernel = torch.flip(t + eps, dims=[-1]) ** (scale - 1)
        fnorm = torch.log(1 + torch.exp(scale * f)) / scale
        u = torch.cumulative_trapezoid(kernel * fnorm, t, dim=-1)
        return u

    return helper


def riel(obs_data, *, model_params, scale, weights, reg_min, t, max_calls, eps):
    return tik_reg(
        f=obs_data,
        model_params=model_params,
        base_loss=transform_loss(
            loss=l1_double, transform=riel_transform(eps=eps, scale=scale, t=t)
        ),
        weights=weights,
        reg_sched=lin_decrease(max_calls=max_calls, _min=reg_min),
    )


def var_tik_frac(
    *,
    obs_data: torch.Tensor,
    alpha: Callable[[int], float],
    weights: Callable[[int], torch.Tensor],
    model_params: torch.nn.Module,
    t: torch.Tensor,
    renorm_scale: float,
    eps: Callable[[int], torch.Tensor]
):
    num_calls = 0

    obs_data_renorm = (
        torch.log(1.0 + torch.exp(renorm_scale * obs_data)) / renorm_scale
    )

    def helper(f):
        nonlocal num_calls, obs_data_renorm
        num_calls += 1
        wts = weights(num_calls)
        frac_exp = alpha(num_calls)

        kernel = ((t + eps(num_calls)) ** frac_exp).flip(dims=[-1])
        fnorm = torch.log(1.0 + torch.exp(renorm_scale * f)) / renorm_scale
        weighted_diff = torch.cumulative_trapezoid(
            kernel * (fnorm - obs_data_renorm), t, dim=-1
        )
        misfit = torch.sum(weighted_diff**2)

        # comeback later and allow for anisotropic penalty -- give tensor for weights[1] instead
        velocity_spatial_grad = (torch.diff(model_params(), dim=-1) ** 2).mean()

        return wts[0] * misfit + wts[1] * velocity_spatial_grad

    return helper


def linear_decrease(*, _min, _max, max_iters):
    def helper(num_calls):
        return _min + (_max - _min) * max(0.0, (1 - num_calls / max_iters))

    return helper


def constant(*, value):
    def helper(num_calls):
        return value

    return helper


def punctuated_decrease(*, _min, _max, period, max_iters):
    num_periods = max_iters // period
    step_size = (_min - _max) / num_periods

    def helper(num_calls):
        epoch = min(num_calls // period, num_periods)
        return _max + step_size * epoch

    return helper


def punctuated_decrease_array(*, _min, _max, period, max_iters):
    protocols = [
        punctuated_decrease(
            _min=_min[i], _max=_max[i], period=period, max_iters=max_iters
        )
        for i in range(len(_min))
    ]

    def helper(num_calls):
        return [protocol(num_calls) for protocol in protocols]

    return helper
