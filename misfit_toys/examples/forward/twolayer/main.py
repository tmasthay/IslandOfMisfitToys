import torch
from misfit_toys.data.dataset import towed_src
import hydra
import deepwave as dw
import matplotlib.pyplot as plt
from masthay_helpers.typlotlib import get_frames_bool, save_frames
from misfit_toys.utils import bool_slice
from masthay_helpers.global_helpers import convert_config_simplest


def out_plotter(*, data, idx, fig, axes, cfg):
    axes.imshow(data[idx].cpu(), **cfg.plot.out.imshow_kw)
    axes.set_title(f'{cfg.plot.out.title}, shot_no={idx[0]}')
    axes.set_xlabel(cfg.plot.out.xlabel)
    axes.set_ylabel(cfg.plot.out.ylabel)

    return {'cfg': cfg}


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg):
    cfg = convert_config_simplest(cfg)

    src_loc = towed_src(**cfg.src).to(cfg.device)

    rec = torch.empty(
        cfg.src.n_shots, cfg.rec.rec_per_shot, 2, device=cfg.device
    )
    rec[:, :, 0] = (
        torch.round(
            torch.linspace(
                cfg.rec.x_min / cfg.dx,
                cfg.rec.x_max / cfg.dx,
                cfg.rec.rec_per_shot,
            )
        )
        .to(dtype=torch.long)
        .to(cfg.device)
    )
    rec[:, :, 1] = (
        torch.round(
            torch.linspace(
                cfg.rec.y_min / cfg.dy,
                cfg.rec.y_max / cfg.dy,
                cfg.rec.rec_per_shot,
            )
        )
        .to(dtype=torch.long)
        .to(cfg.device)
    )

    src_amp = (
        dw.wavelets.ricker(
            cfg.amp.freq, cfg.nt, cfg.dt, cfg.amp.peak_scale / cfg.amp.freq
        )
        .repeat(cfg.src.n_shots, cfg.src.src_per_shot, 1)
        .to(cfg.device)
    )

    v = torch.cat(
        [
            torch.ones(cfg.nx // 2, cfg.ny) * cfg.vp[0],
            torch.ones(cfg.nx // 2, cfg.ny) * cfg.vp[1],
        ]
    ).to(cfg.device)

    out = dw.scalar(
        v,
        cfg.nx,
        cfg.dt,
        source_amplitudes=src_amp,
        source_locations=src_loc,
        receiver_locations=rec,
        accuracy=cfg.accuracy,
        pml_freq=cfg.amp.freq,
    )[-1]

    plt.imshow(v.cpu(), **cfg.plot.v.imshow_kw)
    plt.title(cfg.plot.v.title)
    plt.xlabel(cfg.plot.v.xlabel)
    plt.ylabel(cfg.plot.v.ylabel)
    plt.savefig(cfg.plot.v.name)

    fig, axes = plt.subplots(*cfg.plot.out.shape, figsize=cfg.plot.out.figsize)
    iter = bool_slice(*out.shape, none_dims=(1, 2), ctrl=(lambda x, y: True))
    frames = get_frames_bool(
        data=out, iter=iter, fig=fig, axes=axes, plotter=out_plotter, cfg=cfg
    )
    save_frames(frames, path=cfg.plot.out.name, duration=cfg.plot.out.duration)


if __name__ == "__main__":
    main()
