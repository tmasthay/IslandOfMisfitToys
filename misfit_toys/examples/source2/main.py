import os

import deepwave as dw
import hydra
import matplotlib.pyplot as plt
import torch
import yaml
from mh.core import DotDict, hydra_out, set_print_options, torch_stats
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
        self_key = f'slf_{key.split(".")[-1]}'
        field_name = key.split('.')[-1]
        before_reduction = f'\n\nBefore reduction:\n{pretty_dict(c[key])}'
        c[key] = runtime_reduce(c[key], **resolve_rules, self_key=self_key)
        c[key] = c[key].to(c.device)
        if expected_shape is not None:
            check_shape(c[key], expected_shape, field_name, before_reduction)

    def full_runtime_reduce(lcl_cfg, **kw):
        return runtime_reduce(lcl_cfg, **{**lcl_cfg.resolve, **kw})

    runtime_reduce_simple('data.vp', (c.ny, c.nx))
    input(c.data.vp)
    runtime_reduce_simple('data.rec_loc_y', (c.n_shots, c.rec_per_shot, 2))
    runtime_reduce_simple('data.src_loc_y', (c.n_shots, c.src_per_shot, 2))
    runtime_reduce_simple('data.src_amp_y', (c.n_shots, c.src_per_shot, c.nt))
    # runtime_reduce_simple('data.src_amp_y_init')
    runtime_reduce_simple('data.gbl_rec_loc', None)

    c = full_runtime_reduce(
        c, self_key='slf_gbl_obs_data', call_key="__call_gbl_obs__"
    )
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
    # c = full_runtime_reduce(c, **c.plt.resolve, self_key='slf_plt')
    # for k, v in c.plt.items():
    #     if k not in ['final', 'resolve', 'skip'] + c.plt.get('skip', []):
    #         plt.clf()
    #         v(data=c[f'data.{k}'].detach().cpu())

    if c.get('do_training', True):
        c.data.vp.requires_grad = False
        c.data.curr_src_amp_y = c.data.src_amp_y_init.clone()
        # c.data.curr_src_amp_y = torch.rand(*c.data.src_amp_y_init.shape).to(c.device)
        c.data.curr_src_amp_y.requires_grad = True
        c.train.opt = c.train.opt([c.data.curr_src_amp_y])

        c.results = c.train.loop(c)

        if c.save_tensors:
            # torch.save(
            #     c.data.vp.T.detach().cpu(),
            #     hydra_out('vp.pt'),
            # )
            vp = torch.flip(c.data.vp.T, [0])
            torch.save(vp.detach().cpu(), hydra_out('vp.pt'))
            torch.save(
                c.results.src_amp_frames.detach().cpu(),
                hydra_out('src_amp_frames.pt'),
            )
            torch.save(
                c.data.src_amp_y.squeeze().detach().cpu(),
                hydra_out('true_src_amp_y.pt'),
            )
            torch.save(
                c.results.obs_frames.detach().cpu(), hydra_out('obs_frames.pt')
            )
            torch.save(
                c.data.obs_data.squeeze().detach().cpu(),
                hydra_out('true_obs_data.pt'),
            )
            diff_obs = (
                c.results.obs_frames
                - c.data.obs_data.detach().cpu().unsqueeze(0)
            )
            diff_src = (
                c.results.src_amp_frames
                - c.data.src_amp_y.detach().cpu().unsqueeze(0)
            )
            torch.save(
                diff_obs.detach().cpu().squeeze(), hydra_out('diff_obs_data.pt')
            )
            torch.save(
                diff_src.detach().cpu().squeeze(), hydra_out('diff_src_amp.pt')
            )

        file_path = os.path.dirname(os.path.realpath(__file__))
        os.system(f'cp {file_path}/gen_plot.ipynb {hydra_out()}')

        with open('.latest', 'w') as f:
            f.write(f'cd {hydra_out()}')

        os.makedirs(hydra_out('cfg'), exist_ok=True)
        with open(hydra_out('cfg/plot_cfg.yaml'), 'w') as f:
            yaml.dump(c.plt.dict(), f)

        print('Run following for latest run directory\n        . .latest')


if __name__ == "__main__":
    main()
