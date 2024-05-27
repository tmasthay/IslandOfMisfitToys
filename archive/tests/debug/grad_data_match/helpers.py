import torch


def exp_dist(t, p):
    u = torch.exp(-((t - p[0]) ** 2) / (2 * p[1] ** 2))
    return u / torch.trapz(u, t)
