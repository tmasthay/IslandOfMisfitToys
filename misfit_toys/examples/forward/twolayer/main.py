import deepwave as dw
import hydra
import matplotlib.pyplot as plt
import torch
from mh.core import convert_dictconfig
from mh.typlotlib import get_frames_bool, save_frames

from misfit_toys.data.dataset import towed_src
from misfit_toys.utils import bool_slice


def out_plotter(*, data, idx, fig, axes, cfg):
    plt.clf()
    plt.imshow(data[idx].cpu(), **cfg.plot.out.imshow_kw)
    plt.title(f'{cfg.plot.out.title}, time_step={idx[-1]}')
    plt.xlabel(cfg.plot.out.xlabel)
    plt.ylabel(cfg.plot.out.ylabel)
    plt.colorbar()

    return {'cfg': cfg}


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg):
    cfg = convert_config_simplest(cfg)

    vp = torch.load(cfg.path)[: cfg.nx, : cfg.ny].to(cfg.device)
    src_loc = towed_src(**cfg.src).to(cfg.device)

    x_rec_loc = torch.arange(cfg.rec.padx, cfg.nx - cfg.rec.padx)
    y_rec_loc = torch.arange(cfg.rec.pady, cfg.ny - cfg.rec.pady)
    Y, X = torch.meshgrid(y_rec_loc, x_rec_loc)
    rec = torch.stack([X.flatten(), Y.flatten()], dim=1)
    rec = rec.expand(cfg.src.n_shots, *rec.shape).to(cfg.device)

    src_amp = (
        dw.wavelets.ricker(
            cfg.amp.freq, cfg.nt, cfg.dt, cfg.amp.peak_scale / cfg.amp.freq
        )
        .repeat(cfg.src.n_shots, cfg.src.src_per_shot, 1)
        .to(cfg.device)
    )

    v = torch.cat(
        [
            torch.ones(cfg.ny, cfg.nx // 4) * cfg.vp[0],
            torch.ones(cfg.ny, 3 * cfg.nx // 4) * cfg.vp[1],
        ],
        dim=1,
    ).to(cfg.device)
    v = v.T

    if not cfg.twolayer:
        v = vp

    out = dw.scalar(
        v,
        cfg.dx,
        cfg.dt,
        source_amplitudes=src_amp,
        source_locations=src_loc,
        receiver_locations=rec,
        accuracy=cfg.accuracy,
        pml_freq=cfg.amp.freq,
    )[-1]

    # out = out.reshape(cfg.nx - 2 * cfg.rec.padx, cfg.ny - 2 * cfg.rec.pady, -1)
    # out = out.permute(1, 0, 2)
    out = out.reshape(cfg.ny - 2 * cfg.rec.pady, cfg.nx - 2 * cfg.rec.padx, -1)

    plt.imshow(v.cpu().T, **cfg.plot.v.imshow_kw)
    plt.title(cfg.plot.v.title)
    plt.xlabel(cfg.plot.v.xlabel)
    plt.ylabel(cfg.plot.v.ylabel)
    plt.savefig(cfg.plot.v.name)

    fig, axes = plt.subplots(*cfg.plot.out.shape, figsize=cfg.plot.out.figsize)
    iter = bool_slice(
        *out.shape,
        none_dims=(0, 1),
        ctrl=(lambda x, y: True),
        start=(0, 0, cfg.plot.out.start),
        cut=(0, 0, cfg.plot.out.cut),
        strides=(1, 1, cfg.plot.out.downsample),
    )
    frames = get_frames_bool(
        data=out, iter=iter, fig=fig, axes=axes, plotter=out_plotter, cfg=cfg
    )
    save_frames(frames, path=cfg.plot.out.name, duration=cfg.plot.out.duration)


if __name__ == "__main__":
    main()
