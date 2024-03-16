import logging
import sys

import torch
from mh.core import DotDict

from misfit_toys.fwi.loss.w2 import cum_trap, spline_func, true_quantile
from misfit_toys.utils import taper


class W2Loss(torch.nn.Module):
    def __init__(
        self, *, t, p, obs_data, renorm, gen_deriv, down=1, track=False
    ):
        super().__init__()
        self.org_obs_data = obs_data
        self.obs_data = renorm(obs_data)
        self.renorm = renorm
        self.q_raw = true_quantile(
            self.obs_data, t, p, rtol=0.0, ltol=0.0, err_top=10
        ).to(self.obs_data.device)
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
        loss = torch.trapz(diff**2 * pdf, self.t, dim=-1).sum()
        return loss


class W2LossTracker(torch.nn.Module):
    def __init__(
        self, *, t, p, obs_data, renorm, gen_deriv, down=1, track=False
    ):
        super().__init__()
        self.org_obs_data = obs_data
        self.obs_data = renorm(obs_data)
        self.renorm = renorm
        self.q_raw = true_quantile(
            self.obs_data, t, p, rtol=0.0, ltol=0.0, err_top=10
        ).to(self.obs_data.device)
        self.p = p
        self.t = t
        self.q = spline_func(
            self.p[::down], self.q_raw[..., ::down].unsqueeze(-1)
        )
        self.qd = gen_deriv(q=self.q, p=self.p)
        self.track = track

    def forward(self, traces):
        pdf = self.renorm(traces)
        cdf = cum_trap(pdf, self.t)
        transport = self.q(cdf)
        diff = self.t - transport
        loss = torch.trapz(diff**2 * pdf, self.t, dim=-1).sum()
        if self.track:
            self.pdf = pdf.detach().cpu()
            self.cdf = cdf.detach().cpu()
            self.transport = transport.detach().cpu()
            self.diff = diff.detach().cpu()
        return loss


def relu_renorm(t):
    def helper(x):
        eps = 1.0e-05
        # u = torch.abs(x) + eps
        c = 0.5
        u = torch.exp(c * x) + eps
        v = u / torch.trapz(u, t, dim=-1).unsqueeze(-1)
        if (v < 0).any():
            raise ValueError(f'v is not positive: min={v.min()}')
        return v

    return helper


def softplus(t, eps, beta):
    def helper(x):
        u = torch.log(1.0 + torch.exp(eps * x)) + beta
        v = u / torch.trapz(u, t, dim=-1).unsqueeze(-1)
        return v

    return helper


def hydra_build(c: DotDict, *, down):
    d = DotDict({})
    meta = c.prop.module.meta
    d.obs_data = taper(c.obs_data)
    device = d.obs_data.device
    d.t = torch.linspace(0, meta.dt * meta.nt, meta.nt).to(device)
    d.p = torch.linspace(0, 1, c.np).to(device)
    d.gen_deriv = lambda *args, **kwargs: None
    d.renorm = relu_renorm(d.t)
    d.down = down
    return [], d


def hydra_build_two(*, obs_data, meta, num_probs, down, eps, track, beta):
    d = DotDict({})
    d.obs_data = taper(obs_data)
    device = d.obs_data.device
    d.t = torch.linspace(0, meta.dt * meta.nt, meta.nt).to(device)
    d.p = torch.linspace(0, 1, num_probs).to(device)
    d.gen_deriv = lambda *args, **kwargs: None
    d.renorm = softplus(d.t, eps, beta)
    d.down = down
    d.track = track
    return d


class StdoutLogger(object):
    def __init__(self, logger, level):
        """Initialize with a logger and a log level."""
        self.logger = logger
        self.level = level

    def write(self, message):
        """Write the message to the logger."""
        if message.rstrip() != "":
            self.logger.log(self.level, message.rstrip())

    def flush(self):
        """Flush the stream."""
        pass


def setup_logger():
    logger = logging.getLogger('stdout_logger')
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def main():
    # Set up the logger

    # Now, print statements will go to the logger instead
    print("This is a test message.")


if __name__ == "__main__":
    main()
