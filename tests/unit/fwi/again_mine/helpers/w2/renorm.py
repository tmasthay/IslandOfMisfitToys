from dataclasses import dataclass
from functools import wraps

import torch
import torch.nn.functional as F


@dataclass
class RNames:
    epsilon: float = 1.0e-06


def ensure_renorm(f, *, x, integral=None):
    if integral is None:

        def local_integral(u, t):
            return torch.trapz(u, t, dim=-1)

        integral = local_integral

    @wraps(f)
    def wrapper(*args, **kwargs):
        u = f(*args, **kwargs)
        u = u / integral(u, x)
        return u

    return wrapper


def softplus(*, sharpness=1.0, epsilon=RNames.epsilon):
    @ensure_renorm
    def helper(u):
        return F.softplus(u * sharpness)

    return helper


def absn(epsilon=RNames.epsilon):
    @ensure_renorm
    def helper(u):
        return torch.abs(u) + epsilon

    return helper


def square(epsilon=RNames.epsilon):
    @ensure_renorm
    def helper(u):
        return u**2 + epsilon

    return helper


def add(epsilon=RNames.epsilon):
    @ensure_renorm
    def helper(u):
        return u + epsilon

    return helper


def exponent(sharpness=1.0, epsilon=RNames.epsilon):
    @ensure_renorm
    def helper(u):
        return torch.exp(u * sharpness) + epsilon

    return helper


def sigmoid(sharpness=1.0, epsilon=RNames.epsilon):
    @ensure_renorm
    def helper(u):
        return torch.sigmoid(u * sharpness) + epsilon

    return helper
