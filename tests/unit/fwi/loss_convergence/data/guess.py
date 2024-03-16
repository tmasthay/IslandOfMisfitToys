import torch


def sine(*, x, freq, device):
    return torch.sin(2 * torch.pi * freq * x).to(device)
