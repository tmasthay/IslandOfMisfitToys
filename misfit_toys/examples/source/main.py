import os
from os.path import join as pj

import deepwave as dw
import hydra
import torch
import yaml
from matplotlib import pyplot as plt
from mh.core import DotDict, exec_imports, set_print_options, torch_stats
from mh.typlotlib import get_frames_bool, save_frames
from omegaconf import DictConfig, OmegaConf

from misfit_toys.utils import apply_all

set_print_options(callback=torch_stats('all'))


def preprocess_cfg(cfg: DictConfig):
    c = DotDict(OmegaConf.to_container(cfg, resolve=True))
    c = exec_imports(c)
    c = c.self_ref_resolve()
    c = apply_all(c, relax=True, exc=['rt', 'docs', 'plt'])
    c = apply_all(c, relax=False, exc=['rt', 'docs', 'plt'])
    return c


@hydra.main(config_path="cfg", config_name="cfg", version_base=None)
def main(cfg):
    c: DotDict = preprocess_cfg(cfg)
    c.rt = apply_all(c.rt, relax=False)
    c = c.self_ref_resolve(self_key='slf_obs')
    c.rt = apply_all(c.rt, relax=False, call_key='__call_obs__')
    c = c.self_ref_resolve(self_key='slf_plt')
    apply_all(c.plt, call_key='__plot__', relax=False)

    for k, v in c.plt.items():
        if 'frame_callback' in v:
            try:
                v.frame_callback(data=c.rt[k].detach().cpu())
            except Exception as e:
                v = ValueError(f'{k=}, caused by error above')
                raise v from e

    print(c)


if __name__ == "__main__":
    main()
