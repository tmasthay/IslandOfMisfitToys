from dataclasses import dataclass

import numpy as np
import pytest
import torch
import yaml
from hypothesis import given, settings
from hypothesis import strategies as st
from plot_prob import *
from pytest import mark

from misfit_toys.beta.prob import *

unit_strat = st.floats(min_value=0.0, max_value=1.0, allow_subnormal=False)
# fixed_mu = st.just(0.0)
# fixed_sigma = st.just(1.0)

# curr_strat = {'mu': fixed_mu, 'sigma': fixed_sigma}
curr_strat = {'mu': unit_strat, 'sigma': unit_strat}


@dataclass
class Prob:
    max_examples: int = 10


@mark.fast
@mark.unit
class TestUnbatchSplines:
    @mark.sine
    @given(freq=unit_strat)
    def test_sine_wave_spline_with_random_frequency(self, sine_ref_data, freq):
        x, y, x_test, y_true, y_deriv_true, atol, rtol = sine_ref_data(freq)
        splines = unbatch_splines(x, y)
        for i, spline in enumerate(splines):
            y_pred = spline.evaluate(x_test).squeeze()
            y_deriv_pred = spline.derivative(x_test).squeeze()

            torch.testing.assert_close(y_pred, y_true, atol=atol, rtol=rtol)
            torch.testing.assert_close(
                y_deriv_pred, y_deriv_true, atol=atol, rtol=rtol
            )


@mark.fast
@mark.unit
class TestUnbatchSplinesLambda:
    @mark.sine
    @given(freq=unit_strat)
    def test_sine_wave_spline_with_random_frequency_lambda(
        self, sine_ref_data, freq
    ):
        x, y, x_test, y_true, y_deriv_true, atol, rtol = sine_ref_data(freq)

        F = unbatch_splines_lambda(x, y)
        res = F(x_test).squeeze()
        res_deriv = F(x_test, deriv=True).squeeze()

        torch.testing.assert_close(res, y_true, atol=atol, rtol=rtol)
        torch.testing.assert_close(
            res_deriv, y_deriv_true, atol=atol, rtol=rtol
        )


@mark.fast
@mark.unit
class TestPdf:
    @pytest.fixture(autouse=True)
    def setup(self, cfg, lcl_cfg, report_cfg):
        self.c = lcl_cfg(cfg, 'unit.beta.prob', inherit_keys=['atol', 'rtol'])
        self.cfg = cfg
        self.c.plot = self.c.plot.pdf
        report_cfg(self.c, 'pdf')

    @settings(max_examples=Prob.max_examples, deadline=None)
    @given(**curr_strat)
    def test_analytic_gaussian_pdf(
        self, adjust, mu, sigma, gauss_pdf_computed, gauss_pdf_ref
    ):
        mu = adjust(mu, *self.c.mu)
        sigma = adjust(sigma, *self.c.sigma)
        z, x = gauss_pdf_computed(self.c.x, mu, sigma)
        z_true = gauss_pdf_ref(x, mu, sigma)
        verify_and_plot(
            self,
            plotter=plot_pdf,
            name='pdf',
            computed=z,
            ref=z_true,
            mu=mu,
            sigma=sigma,
            x=x,
        )


@mark.fast
@mark.unit
class TestCdf:
    @pytest.fixture(autouse=True)
    def setup(self, cfg, lcl_cfg, report_cfg):
        self.c = lcl_cfg(cfg, 'unit.beta.prob', inherit_keys=['atol', 'rtol'])
        self.cfg = cfg
        self.c.plot = self.c.plot.cdf
        report_cfg(self.c, 'cdf')

    @settings(max_examples=Prob.max_examples, deadline=None)
    @given(**curr_strat)
    def test_analytic_gaussian_cdf(
        self, adjust, mu, sigma, gauss_pdf_computed, gauss_cdf_ref
    ):
        mu = adjust(mu, *self.c.mu)
        sigma = adjust(sigma, *self.c.sigma)
        z, x = gauss_pdf_computed(self.c.x, mu, sigma)
        z = cdf(z, x, dim=-1)
        z_true = gauss_cdf_ref(x, mu, sigma)
        verify_and_plot(
            self,
            plotter=plot_cdf,
            name='cdf',
            computed=z,
            ref=z_true,
            mu=mu,
            sigma=sigma,
            x=x,
        )


@mark.medium
@mark.unit
class TestDiscQuantile:
    @pytest.fixture(autouse=True)
    def setup(self, cfg, lcl_cfg, report_cfg):
        self.c = lcl_cfg(cfg, 'unit.beta.prob', inherit_keys=['atol', 'rtol'])
        self.cfg = cfg
        self.p = torch.linspace(self.c.p.eps, 1.0 - self.c.p.eps, self.c.p.np)
        self.c.plot = self.c.plot.disc_quantile
        report_cfg(self.c, 'disc_quantile')

    @settings(max_examples=Prob.max_examples, deadline=None)
    @given(**curr_strat)
    def test_analytic_gaussian_disc_quantile(
        self, adjust, mu, sigma, gauss_pdf_computed, gauss_quantile_ref
    ):
        mu = adjust(mu, *self.c.mu)
        sigma = adjust(sigma, *self.c.sigma)
        z, x = gauss_pdf_computed(self.c.x, mu, sigma)
        z = cdf(z, x, dim=-1)
        z = disc_quantile(z, x, p=self.p)
        z_true = gauss_quantile_ref(self.p, mu, sigma)

        # print(f'{mu=}, {sigma=}', flush=True)
        verify_and_plot(
            self,
            plotter=plot_quantile,
            name='disc_quantile',
            computed=z,
            ref=z_true,
            mu=mu,
            sigma=sigma,
            x=x,
        )
