import torch


def ensure_norm(f):
    def helper(u, x, **kwargs):
        v = f(u, **kwargs)
        return v / torch.trapz(v, x, dim=-1).unsqueeze(-1)

    return helper


@ensure_norm
def identity(u):
    return u


@ensure_norm
def abs_renorm(u):
    return torch.abs(u)


@ensure_norm
def square_renorm(u):
    return u**2


@ensure_norm
def softplus(u, *, k=1.0):
    return torch.log(1 + torch.exp(k * u)) / k


@ensure_norm
def relu(u):
    return torch.nn.functional.relu(u)


@ensure_norm
def const(u, *, c):
    return u + c
