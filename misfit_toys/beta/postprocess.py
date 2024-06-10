import os

import torch
import yaml
from mh.core import DotDict


def vp_compare(data: DotDict, *, path: str) -> None:
    vp_true = data.vp_true.reshape(1, -1)
    vp = data.vp.reshape(data.vp.shape[0], -1)
    diff = vp - vp_true

    diff_flat = diff.reshape(diff.shape[0], -1)
    diff_norm = torch.sqrt(torch.mean(diff_flat**2, dim=-1))
    max_diff_norm = torch.min(diff_norm)

    d = dict(l2_diff=max_diff_norm.item())

    filename = os.path.join(path, 'vp_compare.yaml')
    with open(filename, 'w') as f:
        yaml.dump(d, f)
