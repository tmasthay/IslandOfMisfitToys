import hydra
from omegaconf import DictConfig
from mh.core import convert_dictconfig, DotDict, exec_imports, hydra_out
from mh.typlotlib import get_frames_bool, save_frames, apply_subplot
from misfit_toys.utils import bool_slice
from typing import Any
import torch
import matplotlib.pyplot as plt


def preprocess_cfg(cfg: DictConfig) -> DotDict:
    c = convert_dictconfig(cfg)
    c.self_ref_resolve()
    exec_imports(c)

    for k, v in c.plt:
        c[f'{k}.save.path'] = hydra_out(c[f'{k}.save.path'])

    return c


def derive_cfg(c: DotDict) -> DotDict:
    c.slope = torch.linspace(1, 2, 10)
    c.t = torch.linspace(0, 1, 100)
    c.my_template_data = c.t[None, :] * c.slope[:, None]
    return c


def postprocess_cfg(c: DotDict) -> Any:
    fig, axes = plt.subplots(
        *c.plt.plot_name.sub.shape, **c.plt.plot_name.sub.kw
    )
    if c.plt.sub.adjust:
        plt.subplots_adjust(**c.plt.plot_name.sub.adjust)
    iter = bool_slice(*c.my_template_data.shape, **c.plt.plot_name.iter)
    frames = get_frames_bool(
        data=c.my_template_data,
        iter=iter,
        fig=fig,
        axes=axes,
        plotter=plotter,
        c=c,
        framer=None,
    )
    save_frames(frames, **c.plt.plot_name.save)


def plotter(*, data, idx, fig, axes, c):
    plt.clf()
    plt.subplot(*c.plt.plot_name.sub.shape, 1)
    plt.plot(c.t, c.my_template_data[idx], **c.plt.plot_name.my_template_data)
    plt.plot(c.t, c.t, **c.plt.plot_name.identity_layer)
    return {'c': c}


@hydra.main(config_path="cfg", config_name="cfg", version_base=None)
def main(cfg: DictConfig) -> None:
    print(cfg)


if __name__ == "__main__":
    main()
