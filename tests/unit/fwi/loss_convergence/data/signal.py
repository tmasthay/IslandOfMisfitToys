import torch


def gaussian(x, mu, sig, normalize=False):
    u = torch.exp(-((x - mu) ** 2) / (2 * sig**2))
    if normalize:
        u = u / torch.trapz(u, x, dim=-1).unsqueeze(-1)
    return u


def gaussian_mixture(*, x, mus, sigs, weights, normalize=False, device):
    u = torch.zeros_like(x)
    for mu, sig, w in zip(mus, sigs, weights):
        u += w * gaussian(x, mu, sig, normalize)
    if normalize:
        u = u / torch.trapz(u, x, dim=-1).unsqueeze(-1)
    return u.to(device)
