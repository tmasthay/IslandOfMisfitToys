import torch
from returns.curry import curry


@curry
def gaussian(*, x, mu, sigma, device):
    return (torch.randn_like(x) * sigma + mu).to(device)
