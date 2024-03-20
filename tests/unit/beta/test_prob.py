import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from pytest import mark

from misfit_toys.beta.prob import *
from plot_prob import *
from dataclasses import dataclass
import yaml


@mark.fast
@mark.unit
class TestUnbatchSplines:
    @mark.sine
    @given(
        freq=st.floats(
            min_value=0.1, max_value=1.0, exclude_min=True, exclude_max=True
        )
    )
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
    @given(
        freq=st.floats(
            min_value=0.1, max_value=1.0, exclude_min=True, exclude_max=True
        )
    )
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

    @given(
        mu=st.floats(min_value=0.1, max_value=1.0, exclude_min=True),
        sigma=st.floats(min_value=0.1, max_value=1.0, exclude_min=True),
    )
    def test_analytic_gaussian_pdf(
        self, mu, sigma, gauss_pdf_computed, gauss_pdf_ref
    ):
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

    @given(
        mu=st.floats(min_value=0.1, max_value=1.0, exclude_min=True),
        sigma=st.floats(min_value=0.1, max_value=1.0, exclude_min=True),
    )
    def test_analytic_gaussian_cdf(
        self, mu, sigma, gauss_pdf_computed, gauss_cdf_ref
    ):
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

    @settings(max_examples=1)
    @given(
        mu=st.floats(min_value=0.1, max_value=1.0, exclude_min=True),
        sigma=st.floats(min_value=0.1, max_value=1.0, exclude_min=True),
    )
    def test_analytic_gaussian_disc_quantile(
        self, mu, sigma, gauss_pdf_computed, gauss_quantile_ref
    ):
        z, x = gauss_pdf_computed(self.c.x, mu, sigma)

        z = disc_quantile(z, x, p=self.p)
        z_true = gauss_quantile_ref(self.p, mu, sigma)

        verify_and_plot(
            self,
            plotter=plot_quantile,
            name='disc_quantile',
            computed=z,
            ref=z_true,
            mu=mu,
            sigma=sigma,
        )
