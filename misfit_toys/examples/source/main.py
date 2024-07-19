import os
from os.path import join as pj

import deepwave as dw
import hydra
import torch
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
    c = apply_all(c, relax=True, exc=['rt', 'docs'])
    c = apply_all(c, relax=False, exc=['rt', 'docs'])
    return c


@hydra.main(config_path="cfg", config_name="cfg", version_base=None)
def main(cfg):
    c = preprocess_cfg(cfg)
    c.rt = apply_all(c.rt, relax=False)
    print(c)


if __name__ == "__main__":
    main()
