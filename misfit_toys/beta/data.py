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
