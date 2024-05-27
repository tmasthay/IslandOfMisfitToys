import os

import torch


def random_trace(path):
    path = path.replace('conda', os.environ['CONDA_PREFIX'])
    u = torch.load(path)
    u = u.reshape(-1, u.shape[-1])
    idx = torch.randint(0, u.shape[0], (1,)).item()
    while u[idx].norm() < 1e-3:
        idx = torch.randint(0, u.shape[0], (1,)).item()
    return u[idx]


def uniform(x):
    return torch.ones_like(x)


def sine_wave(x):
    return torch.sin(x)


def gaussian(x):
    u = torch.exp(-((x - 0.5) ** 2) / (2 * 0.2**2))
    u = u / torch.trapz(u, x, dim=-1)
    return u


def gaussian_mixture(x, *, mu, sigma, weights):
    u = torch.zeros_like(x)
    for m, s, w in zip(mu, sigma, weights):
        u += w * torch.exp(-((x - m) ** 2) / (2 * s**2))
    u = u / torch.trapz(u, x, dim=-1)
    return u
