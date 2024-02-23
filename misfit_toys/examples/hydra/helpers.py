from misfit_toys.fwi.loss.w2 import true_quantile, spline_func, cum_trap
import torch
from mh.core import DotDict
from misfit_toys.utils import taper


class W2Loss(torch.nn.Module):
    def __init__(self, *, t, p, obs_data, renorm, gen_deriv, down=1):
        super().__init__()
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


def relu_renorm(t):
    def helper(x):
        eps = 1.0e-03
        u = torch.relu(x) + eps
        return u / torch.trapz(u, t, dim=-1).unsqueeze(-1)

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
