from typing import Any

import hydra
import matplotlib.pyplot as plt
import torch
from mh.core import DotDict, convert_dictconfig, draise, exec_imports, hydra_out
from mh.typlotlib import apply_subplot, get_frames_bool, save_frames
from omegaconf import DictConfig, OmegaConf

from misfit_toys.utils import apply_all, bool_slice


def preprocess_cfg(cfg: DictConfig) -> DotDict:
    c = DotDict(OmegaConf.to_container(cfg, resolve=True))
    exec_imports(c)
    c.self_ref_resolve()
    c = apply_all(c, exc=['plotter'])

    return c


def derive_cfg(c: DotDict) -> DotDict:
    return c


def postprocess_cfg(c: DotDict) -> Any:
    print('postprocess')


@hydra.main(config_path="cfg", config_name="cfg", version_base=None)
def main(cfg: DictConfig) -> None:
    c = preprocess_cfg(cfg)
    print(c)


if __name__ == "__main__":
    main()
