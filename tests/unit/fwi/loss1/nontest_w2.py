import os
import pickle

import numpy as np
import pytest
import torch
from hypothesis import given
from hypothesis.strategies import floats, tuples
from mh.core import DotDict, easy_cfg
from plot_w2 import plot_eval
from scipy.interpolate import splev, splrep
from scipy.ndimage import median_filter, uniform_filter

from misfit_toys.fwi.loss.w2 import spline_func, true_quantile, wass2
from misfit_toys.utils import bool_slice
from misfit_toys.utils import mean_filter_1d as mf
from misfit_toys.utils import tensor_summary

c = DotDict(easy_cfg(os.path.join(os.path.dirname(__file__), 'cfg')))


def rerr(s):
    raise ValueError(s)


def rctx(ctx):
    d = {k: v for k, v in ctx.items() if not k.startswith('_')}
    raise ValueError(DotDict(d))


def rtens(ctx, toi=torch.Tensor, reduce=(lambda x: x)):
    d = {}

    def helper(k, v):
        if isinstance(v, toi):
            return {k: reduce(v)}
        elif isinstance(v, dict):
            u = {k: helper(kk, vv) for kk, vv in v.items()}
            u = {k: v for k, v in u.items() if v}
            return u
        else:
            return {}

    for k, v in ctx.items():
        d.update(helper(k, v))

    raise ValueError(DotDict(d))


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
    c.ref_left = c.shift[c.ref_left_idx].item()
    c.ref_scale = c.scale[c.ref_scale_idx].item()

    mask = (c.t >= c.shift[:, None]) & (
        c.t <= c.shift[:, None] + c.scale[:, None, None]
    )
    hat_function = c.eps * torch.exp(
        -((c.t - c.t[len(c.t) // 2]) ** 2) / (2 * c.sig**2)
    )
    uniform_pdfs = mask.float() / c.scale[:, None, None] + hat_function
    uniform_pdfs = uniform_pdfs / torch.trapz(
        uniform_pdfs, dx=c.t[1] - c.t[0], dim=-1
    ).unsqueeze(-1)
    uniform_pdfs = uniform_pdfs.permute(1, 0, 2)
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
        for _ in range(c.filt.deriv.apps):
            w = uniform_filter(w, size=c.filt.deriv.size)
        splines[idx] = torch.from_numpy(v)
        splines_deriv[idx] = torch.from_numpy(w)
    q = spline_func(c.p, splines.unsqueeze(-1))
    qd = spline_func(c.p, splines_deriv.unsqueeze(-1))
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
    bshifts = c.shift[:, None, None]
    bscales = c.scale[None, :, None]
    bt = c.t[None, None, :]
    assert abs(c.ref_left) < c.tol
    ds = bscales - c.ref_scale
    ref_shift = c.shift[len(c.shift) // 2]
    dshift = bshifts - ref_shift
    exp_output = bshifts**2 + bshifts * ds + ds**2 / 3.0
    exp_output = exp_output[..., 0]
    # exp_output = (ds + shifts) ** 2 * 10.0
    torch.save(exp_output, 'exp_output.pt')
    # raise ValueError(f'({ds.min(), ds.max()})')b

    exp_grad = (
        (bt < ref_shift) * (dshift - bt) ** 2
        + bscales * (bshifts + ds)
        + (bt >= ref_shift)
        * (bt <= ref_shift + c.ref_scale)
        * (
            (ds**2 - bscales * ds) * bt**2
            + (2 * bshifts * ds - bscales * bshifts) * bt
            + bshifts**2
            + bscales * bshifts
            + bscales * ds
        )
        + (bt > ref_shift + c.ref_scale) * (dshift - bt) ** 2
    )
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
