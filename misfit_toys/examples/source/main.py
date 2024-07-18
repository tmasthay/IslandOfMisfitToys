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


def create_velocity_model(*, ny, nx, default, piecewise_boxes, smoother):
    v = default * torch.ones(ny, nx)

    def rel2abs(coord, n):
        if type(coord) == int:
            assert 0 <= coord < n, f"coord={coord} is not in bounds [0, {n})"
            return coord
        elif type(coord) == float:
            assert (
                0 <= coord <= 1
            ), f"Relative coord={coord} is not in bounds [0, 1] with n={n}"
            return int(coord * n)
        else:
            raise ValueError(f"coord must be int or float, not {type(coord)}")

    if piecewise_boxes is not None:
        for box in piecewise_boxes:
            y_left, y_right, x_left, x_right, value = box
            y_left = rel2abs(y_left, ny)
            y_right = rel2abs(y_right, ny)
            x_left = rel2abs(x_left, nx)
            x_right = rel2abs(x_right, nx)
            v[y_left:y_right, x_left:x_right] = value
    if smoother is not None:
        v = smoother(v.unsqueeze(0))
    return v


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
