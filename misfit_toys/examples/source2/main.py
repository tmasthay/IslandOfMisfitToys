import os

import deepwave as dw
import hydra
import matplotlib.pyplot as plt
import torch
import yaml
from mh.core import DotDict, set_print_options, torch_stats
from mh.typlotlib import get_frames_bool, save_frames
from omegaconf import OmegaConf

from misfit_toys.utils import exec_imports, runtime_reduce

set_print_options(callback=torch_stats('all'))


def check_shape(data, shape, field, custom_str=''):
    if shape is None:
        return
    if data.shape != shape:
        raise ValueError(
            f"Expected shape {shape} for {field}, got"
            f" {data.shape}\n{custom_str}"
        )


def pretty_dict(d, depth=0, indent_str='  ', s=''):
    for k, v in d.items():
        if isinstance(v, dict) or isinstance(v, DotDict):
            s += f'{indent_str*depth}{k}:\n' + pretty_dict(
                v, depth + 1, indent_str
            )
        else:
            s += f'{indent_str*depth}{k}: {v}\n'
    return s


@hydra.main(config_path="cfg", config_name="cfg", version_base=None)
def main(cfg):
    c = DotDict(OmegaConf.to_container(cfg, resolve=True))
    c = exec_imports(c)

    def runtime_reduce_simple(key, expected_shape=None):
        resolve_rules = c[key].get('resolve', c.resolve)
        field_name = key.split('.')[-1]
        self_key = f'slf_{field_name}'
        # call_key = '__call__'
        field_name = key.split('.')[-1]
        before_reduction = f'\n\nBefore reduction:\n{pretty_dict(c[key])}'
        c[key] = runtime_reduce(c[key], **resolve_rules, self_key=self_key)
        c[key] = c[key].to(c.device)
        if expected_shape is not None:
            check_shape(c[key], expected_shape, field_name, before_reduction)

    def full_runtime_reduce(lcl_cfg, **kw):
        return runtime_reduce(lcl_cfg, **{**lcl_cfg.resolve, **kw})

    c = full_runtime_reduce(c, self_key='slf_pre', call_key='__call_pre__')

    runtime_reduce_simple('data.vp', (c.ny, c.nx))
    runtime_reduce_simple('data.rec_loc_y', (c.n_shots, c.rec_per_shot, 2))
    runtime_reduce_simple('data.src_loc_y', (c.n_shots, c.src_per_shot, 2))
    runtime_reduce_simple('data.src_amp_y', (c.n_shots, c.src_per_shot, c.nt))
    # runtime_reduce_simple('data.src_amp_y_init')
    # runtime_reduce_simple('data.gbl_rec_loc', None)

    # c = full_runtime_reduce(
    #     c, self_key='slf_gbl_obs_data', call_key="__call_gbl_obs__"
    # )
    c = full_runtime_reduce(c, self_key='slf_obs_data', call_key="__call_obs__")
    check_shape(c.data.obs_data, (c.n_shots, c.rec_per_shot, c.nt), 'obs_data')
    c = full_runtime_reduce(
        c, self_key='slf_src_amp_y_init', call_key="__call_src__"
    )
    c = full_runtime_reduce(c, self_key='self', call_key='__call__')
    check_shape(
        c.data.src_amp_y_init,
        (c.n_shots, c.src_per_shot, c.nt),
        'src_amp_y_init',
    )
    c.data.vp.requires_grad = False
    c.data.curr_src_amp_y = c.data.src_amp_y_init.clone()
    c.data.curr_src_amp_y.requires_grad = True
    c.train.opt = c.train.opt([c.data.curr_src_amp_y])

    c.results = c.train.loop(c)

    c = full_runtime_reduce(c, self_key='slf_post', call_key='__call_post__')
    c.post.__rt_callback__(c)


if __name__ == "__main__":
    main()
