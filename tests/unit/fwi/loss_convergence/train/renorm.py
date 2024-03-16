import torch
from returns.curry import curry


def normalize(f, *, t):
    return f / torch.trapz(f, t, dim=-1).unsqueeze(-1)


def softplus(*, t, sharpness=1.0):
    def helper(f):
        u = torch.log(1 + torch.exp(f * sharpness)) / sharpness
        return normalize(u, t=t)

    return helper
