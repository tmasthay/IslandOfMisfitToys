from misfit_toys.fwi.loss.w2 import true_quantile, spline_func, cum_trap
import torch


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

    {
        'plt': {
            'vp': {
                'sub': {
                    'shape': [1, 1],
                    'kw': {'figsize': [10, 10]},
                    'adjust': {},
                },
                'iter': {'none_dims': [-2, -1]},
                'save': {
                    'path': 'figs/vp',
                    'movie_format': 'gif',
                    'duration': 1000,
                },
                'plts': {
                    'vp': {
                        'main': {'opts': {'type': 'imshow', 'cmap': 'seismic'}}
                    }
                },
            }
        },
        'vp': {
            'save': {
                'path': '/home/tyler/Documents/repos/IslandOfMisfitToys/tests/debug/fwi/outputs/2024-02-21/15-14-05/figs/vp'
            }
        },
    }
