import os
from os.path import join as pj

import deepwave as dw
import hydra
import torch
from matplotlib import pyplot as plt
from mh.core import DotDict, exec_imports, set_print_options, torch_stats
from mh.typlotlib import get_frames_bool, save_frames
from omegaconf import DictConfig, OmegaConf
from src_types import *

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
        v = smoother(v)
    return v


def preprocess_cfg(cfg: DictConfig):
    c = DotDict(OmegaConf.to_container(cfg, resolve=True))
    c = exec_imports(c)
    c = c.self_ref_resolve()
    return c


@hydra.main(config_path="cfg", config_name="cfg", version_base=None)
def main(cfg):
    c = preprocess_cfg(cfg)
    v = create_velocity_model(
        ny=c.ny,
        nx=c.nx,
        default=c.vp.default,
        piecewise_boxes=c.vp.piecewise_boxes,
        smoother=c.vp.smoother,
    )
    plt.imshow(v, **c.vp.plt.imshow)
    plt.title(c.vp.plt.title)
    if c.vp.plt.colorbar:
        plt.colorbar()
    plt.savefig(c.vp.plt.save_path)
    print(f"Saved figure to {pj(os.getcwd(), c.vp.plt.save_path)}")


if __name__ == "__main__":
    main()
