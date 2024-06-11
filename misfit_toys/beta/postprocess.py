import datetime
import os
import re
from typing import Callable

import torch
import yaml
from mh.core import DotDict


def get_timestamps(path: str):
    regex = r'HYDRA_TIME_[0-9]{4}-[0-9]{2}-[0-9]{2}/HYDRA_TIME_[0-9]{2}-[0-9]{2}-[0-9]{2}'
    timestamp = re.search(regex, path)
    if not timestamp:
        raise ValueError(f"No timestamp found in {path}")
    timestamp = timestamp.group(0).replace('HYDRA_TIME_', '').replace('/', ' ')
    human_timestamp = datetime.datetime.strptime(timestamp, '%Y-%m-%d %H-%M-%S')
    human_timestamp = human_timestamp.strftime('%B %d, %Y %I:%M:%S %p')

    return {'timestamp': timestamp, 'human_timestamp': human_timestamp}


def core_meta(*, path: str, proj_path: str) -> dict:
    meta = get_timestamps(path)
    meta['proj_path'] = proj_path
    meta['root'] = path
    return meta


def vp_compare(data: DotDict, *, path: str, **kw) -> None:
    vp_true = data.vp_true.reshape(1, -1)
    vp = data.vp.reshape(data.vp.shape[0], -1)
    diff = vp - vp_true

    diff_flat = diff.reshape(diff.shape[0], -1)
    diff_norm = torch.sqrt(torch.mean(diff_flat**2, dim=-1))
    max_diff_norm = torch.min(diff_norm)

    d = core_meta(path=path, **kw)
    d['l2_diff'] = max_diff_norm.item()

    filename = os.path.join(path, 'vp_compare.yaml')
    with open(filename, 'w') as f:
        yaml.dump(d, f)
