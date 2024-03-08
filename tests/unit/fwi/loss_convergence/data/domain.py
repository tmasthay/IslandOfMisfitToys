import torch


def gen_domain(*args, device, **kwargs):
    return torch.linspace(*args, **kwargs).to(device)


def gen_probs(*, t, upsample, device):
    return torch.linspace(0, 1, t.shape[-1] * upsample).to(device)
