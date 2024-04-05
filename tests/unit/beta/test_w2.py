from dataclasses import dataclass

import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

# import plot_prob
import plot_w2
import pytest
from pytest import mark

from misfit_toys.beta.prob import cdf, pdf
from misfit_toys.beta.w2 import w2

unit_strat = st.floats(min_value=0.0, max_value=1.0, allow_subnormal=False)
curr_strat = {
    'mu1': unit_strat,
    'sigma1': unit_strat,
    'mu2': unit_strat,
    'sigma2': unit_strat,
}


@dataclass
class W2Local:
    max_examples: int = 1


@mark.fast
@mark.unit
@mark.wass
class TestW2:
    @pytest.fixture(autouse=True)
    def setup(self, cfg, lcl_cfg, report_cfg):
        self.c = lcl_cfg(cfg, 'unit.beta.w2', inherit_keys=['atol', 'rtol'])
        self.cfg = cfg
        self.c.plot = self.c.plot.w2
        self.p = self.c.p
        report_cfg(self.c, 'w2')

    @settings(max_examples=W2Local.max_examples, deadline=None)
    @given(**curr_strat)
    def test_gaussian_w2(
        self,
        adjust,
        mu1,
        sigma1,
        mu2,
        sigma2,
        gauss_pdf_computed,
        gauss_quantile_ref,
    ):
        mu1, mu2 = adjust(mu1, *self.c.mu1), adjust(mu2, *self.c.mu2)
        sigma1, sigma2 = adjust(sigma1, *self.c.sigma1), adjust(
            sigma2, *self.c.sigma2
        )
        if sigma1 < 0.1 or sigma2 < 0.1:
            raise ValueError('sigma1 and sigma2 must be greater than 0.1')
        z1, x1 = gauss_pdf_computed(self.c.x, mu1, sigma1)
        z2, x2 = gauss_pdf_computed(self.c.x, mu2, sigma2)

        w2_func, Q = w2(z1, renorm=self.c.renorm, x=x1, p=self.p)

        w2_val = w2_func(z2)
        true_val = torch.tensor((mu2 - mu1) ** 2 + (sigma2 - sigma1) ** 2)

        CDF = cdf(z2, x2, dim=-1)
        Tcomputed = Q(CDF, deriv=False).squeeze() - x2
        Tref = mu1 + sigma1 / sigma2 * (x2 - mu2) - x2
        Tcomputed = Tcomputed**2 * z1
        Tref = Tref**2 * z1

        Qcomputed = Q(self.p, deriv=False).squeeze()
        Qref = gauss_quantile_ref(self.p, mu1, sigma1)

        cdfComputed = CDF
        cdfRef = 0.5 * (1 + torch.erf((x2 - mu2) / (sigma2 * np.sqrt(2))))

        plot_w2.verify_and_plot(
            self,
            plotter=plot_w2.plot_w2,
            name='w2',
            computed=w2_val,
            ref=true_val,
            mu1=mu1,
            sigma1=sigma1,
            mu2=mu2,
            sigma2=sigma2,
            x=x1,
            Qref=Qref,
            Qcomputed=Qcomputed,
            Tref=Tref,
            Tcomputed=Tcomputed,
            cdfComputed=cdfComputed,
            cdfRef=cdfRef,
            exclusions=[
                'Qref',
                'Qcomputed',
                'Tref',
                'Tcomputed',
                'cdfComputed',
                'cdfRef',
            ],
        )
        # return True
