import os

import hydra
import matplotlib.pyplot as plt
import torch
from hydra import utils
from mh.core import DotDict, convert_dictconfig, hydra_out, set_print_options
from mh.typlotlib import get_frames_bool, save_frames
from omegaconf import DictConfig
from scipy.interpolate import splev, splrep
from torch.nn import MSELoss

from misfit_toys.fwi.loss.w2 import cum_trap, spline_func, true_quantile
from misfit_toys.utils import bool_slice
from misfit_toys.utils import mean_filter_1d as mf

set_print_options(precision=3, threshold=10)


class W2Loss(torch.nn.Module):
    def __init__(self, *, t, p, obs_data, renorm, gen_deriv, down=1):
        super().__init__()
        self.obs_data = renorm(obs_data)
        self.renorm = renorm
        self.q_raw = true_quantile(
            self.obs_data, t, p, rtol=0.0, ltol=0.0, err_top=10
        )
        self.p = p
        self.t = t
        self.q = spline_func(
            self.p[::down], self.q_raw[..., ::down].unsqueeze(-1)
        )
        self.qd = gen_deriv(q=self.q, p=self.p)

    def forward(self, traces):
        pdf = self.renorm(traces)
        cdf = cum_trap(pdf, self.t)
        transport = self.q(cdf)
        diff = self.t - transport
        loss = torch.trapz(diff**2 * pdf, self.t, dim=-1)
        return loss


class MSEOpt(torch.nn.Module):
    def __init__(self, t, pdf_true, pdf):
        super().__init__()
        self.loss_fn = MSELoss()
        self.t = t
        self.pdf_true = pdf_true
        self.pdf = pdf
        self.renorm = lambda x: x

    def forward(self, pdf):
        loss = self.loss_fn(pdf, self.pdf_true)
        return loss


class W2Opt(torch.nn.Module):
    def __init__(self, *, t, p, pdf_true, pdf, renorm, gen_deriv):
        super().__init__()
        self.loss_fn = W2Loss(
            t=t, p=p, obs_data=pdf_true, renorm=renorm, gen_deriv=gen_deriv
        )
        self.pdf = pdf
        self.renorm = renorm

    def forward(self, pdf):
        loss = self.loss_fn(pdf)
        return loss


class Paths:
    path = "cfg"
    name = "cfg"


def exp_dist(t, p):
    u = torch.exp(-((t - p[0]) ** 2) / (2 * p[1] ** 2))
    return u / torch.trapz(u, t)


def preprocess_cfg(cfg: DictConfig) -> DotDict:
    c = convert_dictconfig(cfg)
    c.t = torch.linspace(*c.t)
    c.p = torch.linspace(*c.p)
    c.plt.gauss.save.path = hydra_out(c.plt.gauss.save.path)
    return c


def derived_cfg(c: DotDict) -> DotDict:
    def renorm(f):
        u = f + c.eps
        return u / torch.trapz(u, c.t)

    def gen_deriv(*, q, p):
        u = q(p, deriv=True)
        for _ in range(10):
            u = mf(u, 10)
        qd = spline_func(p, u.unsqueeze(-1))
        return qd

    dm = c.derived_meta
    c.pdf_true = torch.zeros_like(c.t)
    c.pdf = torch.zeros_like(c.t)
    for i in range(len(c.param0)):
        c.pdf_true += exp_dist(c.t, c.param0[i])
    for i in range(len(c.param)):
        c.pdf += exp_dist(c.t, c.param[i])
    c.pdf_true = renorm(c.pdf_true) + c.eta * torch.randn_like(c.t)
    c.pdf = torch.nn.Parameter(renorm(c.pdf))

    c.history = [c.pdf.clone().detach().cpu()]

    c.gen_deriv = gen_deriv
    c.renorm = renorm
    c.pdf_true = renorm(c.pdf_true)
    c.w2_model = W2Opt(
        t=c.t,
        p=c.p,
        pdf_true=c.pdf_true,
        pdf=c.pdf,
        renorm=c.renorm,
        gen_deriv=c.gen_deriv,
    )
    c.mse_model = MSEOpt(t=c.t, pdf_true=c.pdf_true, pdf=c.pdf)
    c.model = c[dm.models[dm.chosen].key].to(c.device)
    c.loss_history = [c.model(c.pdf).detach().item()]
    return c


def optimize(c: DotDict) -> DotDict:
    optimizer = torch.optim.SGD(c.model.parameters(), lr=c.opt.lr)
    for i in range(c.opt.max_iter):
        optimizer.zero_grad()
        loss = c.model(c.pdf)
        loss.backward()
        optimizer.step()
        c.history.append(c.model.renorm(c.pdf.clone().detach().cpu()))
        c.loss_history.append(loss.detach().item())
        print(f'iter={i}, loss={c.loss_history[-1]}', end='\r')
        if loss.item() < c.opt.tol:
            break
    c.history = torch.stack(c.history, dim=0)
    return c


def plotter(*, data, idx, fig, axes, c):
    plt.clf()
    plt.subplot(2, 1, 1)
    plt.plot(c.t, c.pdf_true, **c.plt.gauss.ref)
    plt.plot(c.t, c.history[idx], **c.plt.gauss.pred)
    plt.title(f"PDF at iter {idx}")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(c.loss_history)
    plt.plot([idx[0]], [c.loss_history[idx[0]]], "ro")
    plt.title("Loss")
    return {"c": c}


def postprocess_cfg(c: DotDict) -> DotDict:
    fig, axes = plt.subplots(*c.plt.sub.shape, **c.plt.sub.kw)
    iter = bool_slice(*c.history.shape, **c.plt.iter)
    frames = get_frames_bool(
        data=None, iter=iter, fig=fig, axes=axes, plotter=plotter, c=c
    )
    save_frames(frames, **c.plt.gauss.save)


@hydra.main(config_path=Paths.path, config_name=Paths.name, version_base=None)
def main(cfg: DictConfig) -> None:
    c = preprocess_cfg(cfg)
    c = derived_cfg(c)
    c = optimize(c)
    postprocess_cfg(c)

    print(f"Done...plots in {c.plt.gauss.save.path}.gif")


if __name__ == "__main__":
    main()
