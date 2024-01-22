import torch
from misfit_toys.data.dataset import towed_src
import hydra
import deepwave as dw
import matplotlib.pyplot as plt


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg):
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
            torch.ones(cfg.nx // 2, cfg.ny // 2) * cfg.vp[0],
            torch.ones(cfg.nx // 2, cfg.ny // 2) * cfg.vp[1],
        ]
    ).to(cfg.device)

    plt.imshow(v.cpu(), **cfg.plot.v.imshow_kw)
    plt.title(cfg.plot.v.title)
    plt.xlabel(cfg.plot.v.xlabel)
    plt.ylabel(cfg.plot.v.ylabel)
    plt.savefig(cfg.plot.v.name)


if __name__ == "__main__":
    main()
