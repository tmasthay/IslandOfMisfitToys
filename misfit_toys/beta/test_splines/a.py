import hydra
import matplotlib.pyplot as plt
import torch
import yaml
from mh.core import DotDict, convert_dictconfig, draise, torch_stats
from mh.typlotlib import get_frames_bool, save_frames
from omegaconf import DictConfig
from torchcubicspline import NaturalCubicSpline as Spline
from torchcubicspline import natural_cubic_spline_coeffs as ncs

from misfit_toys.utils import bool_slice, clean_idx

torch.set_printoptions(
    precision=3, sci_mode=True, threshold=5, callback=torch_stats('all')
)


def get_linspace(v):
    return torch.linspace(*v.args, **v.kw)


def preprocess_cfg(cfg: DictConfig) -> DotDict:
    c = convert_dictconfig(cfg)
    c.p = get_linspace(c.p)
    c.mu = get_linspace(c.mu)
    c.sigma = get_linspace(c.sigma)
    return c


def make_plots(c: DotDict) -> None:
    def plotter(*, data, idx, fig, axes):
        plt.clf()
        plt.plot(data.p, data.q[idx])
        plt.ylim([c.q.min(), c.q.max()])
        plt.title(clean_idx(idx))
        plt.tight_layout()

    fig, axes = plt.subplots(*c.plt.sub.shape, **c.plt.sub.kw)
    iter = bool_slice(*c.q.shape, **c.plt.iter)
    frames = get_frames_bool(
        data=c, iter=iter, fig=fig, axes=axes, plotter=plotter
    )
    save_frames(frames, **c.plt.save)
    print(f"Saved to {c.plt.save.path}")


@hydra.main(config_path='cfg', config_name='cfg', version_base=None)
def main(cfg: DictConfig) -> None:
    c = preprocess_cfg(cfg)

    s2 = 1.4142135623730951
    # c.pdf = torch.exp(())
    c.q = c.mu[None, :, None] + s2 * c.sigma[None, None, :] * torch.erfinv(
        2 * c.p[:, None, None] - 1.0
    )
    c.q = c.q.unsqueeze(0)
    # make_plots(c)
    print(c.q)


if __name__ == "__main__":
    main()
