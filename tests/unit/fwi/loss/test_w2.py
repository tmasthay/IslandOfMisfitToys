import os
import pickle

import numpy as np
import pytest
import torch
from hypothesis import given
from hypothesis.strategies import floats, tuples
from masthay_helpers.global_helpers import DotDict, easy_cfg
from plot_w2 import plot_eval
from scipy.interpolate import splev, splrep
from scipy.ndimage import median_filter, uniform_filter

from misfit_toys.fwi.loss.w2 import (
    quantile_deriv,
    spline_func,
    true_quantile,
    wass2,
)
from misfit_toys.utils import bool_slice
from misfit_toys.utils import mean_filter_1d as mf
from misfit_toys.utils import tensor_summary

c = DotDict(easy_cfg(os.path.join(os.path.dirname(__file__), 'cfg')))


def dnumpy(f):
    def helper(x, *args, **kwargs):
        return torch.from_numpy(f(x.numpy(), *args, **kwargs))

    return helper


def get_filter(name, size, apps, clamp=None):
    def helper(x, t=None):
        if clamp is not None:
            x = torch.clamp(x, *clamp)
        if name == 'median':
            u = median_filter(x.numpy(), size=size)
            for _ in range(apps - 1):
                u = median_filter(u, size=size)
            return torch.from_numpy(u)
        elif name == 'mean':
            u = uniform_filter(x.numpy(), size=size)
            for _ in range(apps - 1):
                u = uniform_filter(u, size=size)
            return torch.from_numpy(u)
        elif name in ['approx', 'spline_approx']:
            tck = splrep(t.numpy()[::4], x.numpy()[::4], k=3, s=10)
            y = splev(t.numpy(), tck)
            return torch.from_numpy(y)
        else:
            return x

    return helper


@pytest.fixture(scope='session')
def ref():
    c.set('tp', c.t)
    c.set('shiftp', c.shift)
    c.set('scalep', c.scale)

    c.t = torch.linspace(c.t[0], c.t[1], c.t[2])
    c.p = torch.linspace(0.0, 1.0, c.np)

    c.shift = torch.linspace(c.shift[0], c.shift[1], c.shift[2])
    c.scale = torch.linspace(c.scale[0], c.scale[1], c.scale[2])

    c.ref_left_idx = int(c.ref_idx[0] * c.shift.shape[0])
    c.ref_scale_idx = int(c.ref_idx[1] * c.scale.shape[0])
    # assert c.ref_left_idx == 1000
    # assert c.ref_scale_idx == 1000
    # assert c.shift == 100
    c.ref_left = c.shift[c.ref_left_idx].item()
    c.ref_scale = c.scale[c.ref_scale_idx].item()

    mask = (c.t >= c.shift[:, None]) & (
        c.t <= c.shift[:, None] + c.scale[:, None, None]
    )
    uniform_pdfs = mask.float() / c.scale[:, None, None]
    uniform_pdfs[:, :, : c.neps] = c.eps
    uniform_pdfs[:, :, -(1 + c.neps) :] = c.eps
    uniform_pdfs = uniform_pdfs / torch.trapz(
        uniform_pdfs, dx=c.t[1] - c.t[0], dim=-1
    ).unsqueeze(-1)
    uniform_pdfs = uniform_pdfs.permute(1, 0, 2)
    # raise ValueError(u.shape)
    # uniform_pdfs = mask.float() / c.scale[:, None, None]
    # uniform_pdfs = uniform_pdfs.permute(1, 0, 2)
    ref_pdf = uniform_pdfs[c.ref_left_idx, c.ref_scale_idx, :]
    uniform_pdfs /= torch.trapezoid(
        uniform_pdfs, dx=c.t[1] - c.t[0], dim=-1
    ).unsqueeze(-1)
    # deriv_filter_func = get_filter(**c.filt.deriv)
    # filter_func = get_filter(**c.filt.eval)

    Q = true_quantile(
        uniform_pdfs,
        c.t,
        c.p,
        ltol=c.ltol,
        rtol=c.rtol,
        atol=c.atol,
        err_top=c.err_top,
    )
    splines = torch.empty(Q.shape)
    splines_deriv = torch.empty(Q.shape)
    for idx, _ in bool_slice(*Q.shape, none_dims=[2]):
        weights = np.ones_like(c.p.numpy())
        left_cutoff = int(c.decay.frac * len(c.p))
        right_cutoff = len(c.p) - left_cutoff
        if left_cutoff > 0:
            weights[:left_cutoff] = (
                np.array(list(range(left_cutoff))) / left_cutoff
            ) ** c.decay.rate
            weights[right_cutoff:] = (
                np.flip(weights[:left_cutoff])
            ) ** c.decay.rate
        # weights = np.abs(np.array(range(len(c.p))) - len(c.p) / 2)
        weights = weights[:: c.downsample]
        u = splrep(
            c.p.numpy()[:: c.downsample],
            Q[idx].numpy()[:: c.downsample],
            k=c.k,
            s=c.smooth,
            w=weights,
        )
        v = splev(c.p.numpy(), u)
        w = splev(c.p.numpy(), u, der=1)
        splines[idx] = torch.from_numpy(v)
        splines_deriv[idx] = torch.from_numpy(w)

    q = spline_func(c.p, splines.unsqueeze(-1))
    qd = spline_func(c.p, splines_deriv.unsqueeze(-1))

    # q, qd = quantile_deriv(
    #     uniform_pdfs,
    #     c.t,
    #     c.p,
    #     deriv_filter_func=deriv_filter_func,
    #     filter_func=filter_func,
    #     ltol=c.ltol,
    #     rtol=c.rtol,
    #     atol=c.atol,
    #     err_top=c.err_top,
    # )
    f = wass2(q, qd)
    return f, q, qd, uniform_pdfs, ref_pdf


@pytest.fixture(scope='session')
def evaluation(ref):
    # check array indexing
    w2, q, qd, uniform_pdfs, ref_pdf = ref
    misfit = w2.eval(pdf=ref_pdf, t=c.t)
    grad = w2.grad(
        pdf=ref_pdf, t=c.t, cdf=misfit.cdf, transport=misfit.transport
    )
    return DotDict(
        {
            'w2': w2,
            'q': q,
            'qd': qd,
            'uniform_pdfs': uniform_pdfs,
            'ref_pdf': ref_pdf,
            'misfit': misfit,
            'grad': grad,
        }
    )


def test_shapes(evaluation):
    d = evaluation
    assert list(d.uniform_pdfs.shape) == [c.shiftp[2], c.scalep[2], c.tp[2]]
    assert list(d.q(d.ref_pdf).shape[:-1]) == [c.shiftp[2], c.scalep[2]]
    assert list(d.qd(d.ref_pdf).shape[:-1]) == [c.shiftp[2], c.scalep[2]]
    assert list(d.ref_pdf.shape) == [c.tp[2]]
    assert list(d.misfit.res.shape) == [c.shiftp[2], c.scalep[2]]
    assert list(d.grad.shape) == [c.shiftp[2], c.scalep[2], c.tp[2]]


def test_eval(evaluation):
    d = evaluation
    shifts, scales = c.shift[:, None], c.scale[None, :]
    # raise ValueError(shifts)
    assert abs(c.ref_left) < c.tol
    # assert abs(c.ref_scale - 1.0) < c.tol
    ds = scales - c.ref_scale
    # alpha = ds / c.ref_scale
    # beta = shifts / c.ref_scale
    # gamma = ds + beta
    exp_output = shifts**2 + shifts * ds + ds**2 / 3.0
    # exp_output = (ds + shifts) ** 2 * 10.0
    torch.save(exp_output, 'exp_output.pt')
    # raise ValueError(f'({ds.min(), ds.max()})')
    exp_grad = None
    if c.plt.eval:
        plot_eval(
            d,
            exp_output,
            exp_grad,
            config_path=c.plt.path,
            config_name='eval',
            pytest_cfg=c,
        )
    assert torch.allclose(
        d.misfit.res,
        exp_output,
        atol=c.test_tol.eval.atol,
        rtol=c.test_tol.eval.rtol,
    ), tensor_summary(d.misfit.res - exp_output)


if __name__ == '__main__':
    pytest.main()
