import torch
import numpy as np
import matplotlib.pyplot as plt
from mh.core import DotDict, draise
from misfit_toys.utils import bool_slice
from mh.typlotlib import get_frames_bool, save_frames
import os
from misfit_toys.beta.helpers import pdf, cdf, cts_quantile, combos
import numpy as np


def direct_quantile(plotter):
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

    Q = cts_quantile(CDF, x, p=p)
    Qcomp = Q(p).squeeze().detach()

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

    out_file = os.path.join(os.getcwd(), 'prob4.gif')
    out_file = os.path.abspath(out_file)
    print(f'\nSee {out_file}\n')

    return d


def ref_quantile(plotter):
    x = torch.linspace(-5, 5, 100)
    eps = 0.01
    p = torch.linspace(eps, 1.0 - eps, 100)
    mu = torch.linspace(-2, 2, 30)
    sig = torch.linspace(0.5, 1.5, 30)
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

    PDF_flat = PDF.view(-1, PDF.shape[-1])
    PDF_mid = PDF_flat[PDF_flat.shape[0] // 2, :]
    CDF_flat = CDF.view(-1, CDF.shape[-1])
    CDF_mid = CDF_flat[CDF_flat.shape[0] // 2, :]
    all = combos(mu, sig)
    param_ref = all[all.shape[0] // 2, :]

    Q = cts_quantile(CDF, x, p=p, tol=1.0e-04, max_iters=20)
    # Qcomp = Q(p).squeeze().detach()
    Qcomp = Q(CDF_mid).squeeze().detach()
    Qcomp_ref = all[:, 0, None] + all[:, 1, None] / param_ref[1] * (
        x[None, :] - param_ref[0]
    )
    Qcomp_ref = Qcomp_ref.view(Qcomp.shape)

    # draise(Qcomp.shape, Qcomp_ref.shape, PDF_mid.shape, CDF_mid.shape, x.shape)
    integrands = (Qcomp - x[None, None, :]) ** 2 * PDF_mid[None, None, :]
    distances = torch.trapz(integrands, x, dim=-1)
    distances_ref = (mu[:, None] - param_ref[0]) ** 2 + (
        sig[None, :] - param_ref[1]
    ) ** 2
    # draise(distances)

    # draise(Qcomp, Qcomp_ref)

    d = DotDict(
        dict(
            x=x,
            p=p,
            raw=PDF,
            cdf=CDF,
            Q=Qcomp,
            Qref=Qcomp_ref,
            mu=mu,
            sig=sig,
            cdf_ref=cdf_ref,
            pdf_ref=pdf_ref,
            distances=distances,
            distances_ref=distances_ref,
            distances_diff=torch.abs(distances - distances_ref),
        )
    )
    num_frames = 20
    total = Qcomp.shape[0] * Qcomp.shape[1]
    alpha = max(1, int(np.sqrt(total // num_frames)))
    strides = [alpha, alpha, 1]
    shape = (3, 2)
    fig, axes = plt.subplots(*shape, figsize=(10, 10))
    iter = bool_slice(*Qcomp.shape, none_dims=[-1], strides=strides)
    frames = get_frames_bool(
        iter=iter,
        plotter=plotter,
        data=d,
        shape=shape,
        fig=fig,
        axes=axes,
    )
    save_frames(frames, path='prob4.gif', duration=1000)

    out_file = os.path.join(os.getcwd(), 'prob4.gif')
    out_file = os.path.abspath(out_file)
    print(f'\nSee {out_file}\n')

    return d
