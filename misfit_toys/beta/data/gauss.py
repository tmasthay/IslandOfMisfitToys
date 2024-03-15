import torch
import numpy as np
import matplotlib.pyplot as plt
from mh.core import DotDict
from misfit_toys.utils import bool_slice
from mh.typlotlib import get_frames_bool, save_frames
import os
from ..helpers import pdf, cdf, cts_quantile


def main4():
    x = torch.linspace(-5, 5, 25)
    eps = 0.01
    p = torch.linspace(eps, 1.0 - eps, 100)
    mu = torch.linspace(-2, 2, 3)
    sig = torch.linspace(0.5, 1.5, 7)
    u = torch.exp(
        -((x[None, None, :] - mu[:, None, None]) ** 2)
        / (2 * sig[None, :, None] ** 2)
    )

    pdf_ref = u * (1 / (sig[None, :, None] * np.sqrt(2 * np.pi)))
    cdf_ref = 0.5 * (
        1.0
        + torch.erf(
            (x[None, None, :] - mu[:, None, None])
            / (sig[None, :, None] * np.sqrt(2))
        )
    )
    Qref = mu[:, None, None] + sig[None, :, None] * np.sqrt(2) * torch.erfinv(
        2 * p - 1
    )

    def renorm(v, y):
        return v / torch.trapz(v, y, dim=-1).unsqueeze(-1)

    PDF = pdf(u, x, renorm=renorm)
    CDF = cdf(PDF, x)
    # Q = get_quantile_lambda(u, x, p=p, renorm=renorm)
    Q = cts_quantile(CDF, x, p=p)
    # Q = Q.reshape(int(np.prod(Q.shape)))
    # Qcomp = torch.Tensor(np.array([e.evaluate(p) for e in Q])).squeeze()
    # Qcomp = Qcomp.view(mu.numel(), sig.numel(), -1)
    # input(Qcomp.shape)
    Qcomp = Q(p).squeeze().detach()
    # Qcomp = disc_quantile(CDF, x, p=p)
    # Qref = [norm.ppf(p, loc=m, scale=s) for m, s in combos(mu, sig, flat=True)]
    # Qref = np.array(Qref).reshape(*Qcomp.shape)

    def plotter(*, data, idx, fig, axes, shape):
        plt.clf()
        plt.suptitle(
            r'$\mu={:.2f}, \sigma={:.2f}$'.format(
                data.mu[idx[0]], data.sig[idx[1]]
            )
        )
        plt.subplot(*shape, 1)
        plt.plot(data.x, data.raw[idx], label='Computed PDF')
        plt.plot(
            data.x, data.pdf_ref[idx], 'ro', label='Analytic PDF', markersize=1
        )
        plt.ylim(data.raw.min(), data.raw.max())
        plt.title('PDF')

        plt.subplot(*shape, 2)
        plt.plot(data.x, data.cdf[idx], label='Computed CDF')
        plt.plot(
            data.x, data.cdf_ref[idx], 'ro', label='Analytic CDF', markersize=1
        )
        plt.title('CDF')

        plt.subplot(*shape, 3)
        plt.plot(data.p, data.Q[idx], label=r'$Q$')
        plt.plot(
            data.p, data.Qref[idx], 'ro', markersize=1, label=r'Analytic $Q$'
        )
        plt.legend(framealpha=0.3)
        plt.title('Quantile')
        plt.ylim(data.x.min(), data.x.max())
        plt.tight_layout()
        return {'shape': shape}

    d = DotDict(
        dict(
            x=x,
            p=p,
            raw=PDF,
            cdf=CDF,
            Q=Qcomp,
            Qref=Qref,
            mu=mu,
            sig=sig,
            cdf_ref=cdf_ref,
            pdf_ref=pdf_ref,
        )
    )
    shape = (3, 1)
    fig, axes = plt.subplots(*shape, figsize=(10, 10))
    iter = bool_slice(*Qcomp.shape, none_dims=[-1])
    frames = get_frames_bool(
        iter=iter,
        plotter=plotter,
        data=d,
        shape=shape,
        fig=fig,
        axes=axes,
    )
    save_frames(frames, path='prob4.gif', duration=1000)

    out_file = os.path.join(__file__, '..', 'prob4.gif')
    out_file = os.path.abspath(out_file)
    print(f'\nSee {out_file}\n')

    return d
