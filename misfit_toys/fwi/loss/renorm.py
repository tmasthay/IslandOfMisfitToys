import torch


def square(x):
    u = x**2
    return u / u.sum()


def shift(x, s=None):
    if s is None:
        s = x.min()
    u = x - s
    return u / u.sum()


def exp(x, alpha=-1.0):
    u = torch.exp(alpha * x)
    return u / u.sum()
