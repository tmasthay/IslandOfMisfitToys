import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from pytest import mark

from misfit_toys.beta.prob import *


@dataclass
class Prob:
    max_examples: int = 5


@mark.fast
@mark.interpolate
class TestUnbatchSplines:
    @mark.sine
    @given(
        freq=st.floats(
            min_value=0.1, max_value=1.0, exclude_min=True, exclude_max=True
        )
    )
    @settings(max_examples=Prob.max_examples)
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
@mark.interpolate
class TestUnbatchSplinesLambda:
    @mark.sine
    @given(
        freq=st.floats(
            min_value=0.1, max_value=1.0, exclude_min=True, exclude_max=True
        )
    )
    @settings(max_examples=Prob.max_examples)
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
@mark.prob
class TestPdf:
    @mark.cfg
    @pytest.fixture(autouse=True)
    def setup(self, cfg):
        self.c = cfg.unit.beta.pdf
        self.cfg = cfg
        self.c.atol = self.c.get("atol", cfg.atol)
        self.c.rtol = self.c.get("rtol", cfg.rtol)

    @mark.gaussian
    @given(
        mu=st.floats(min_value=0.1, max_value=1.0, exclude_min=True),
        sigma=st.floats(min_value=0.1, max_value=1.0, exclude_min=True),
    )
    @settings(max_examples=Prob.max_examples)
    def test_analytic_gaussian_pdf(self, mu, sigma):
        x = torch.linspace(*self.c.x)
        z = torch.exp(-((x - mu) ** 2) / (2 * sigma**2))

        def renorm(y1, x1):
            return y1 / torch.trapz(y1, x1, dim=-1).unsqueeze(-1)

        z = pdf(z, x, renorm=renorm)
        z_true = (
            1
            / (sigma * np.sqrt(2 * np.pi))
            * torch.exp(-((x - mu) ** 2) / (2 * sigma**2))
        )
        torch.testing.assert_close(
            z, z_true, rtol=self.c.atol, atol=self.c.rtol
        )


@mark.fast
class TestCdf:
    @mark.cfg
    @pytest.fixture(autouse=True)
    def setup(self, cfg):
        self.c = cfg.unit.beta.pdf
        self.cfg = cfg
        self.c.atol = self.c.get("atol", cfg.atol)
        self.c.rtol = self.c.get("rtol", cfg.rtol)

    @mark.gaussian
    @given(
        mu=st.floats(min_value=0.1, max_value=1.0, exclude_min=True),
        sigma=st.floats(min_value=0.1, max_value=1.0, exclude_min=True),
    )
    @settings(max_examples=Prob.max_examples)
    def test_analytic_gaussian_cdf(self, mu, sigma):
        x = torch.linspace(*self.c.x)
        z = torch.exp(-((x - mu) ** 2) / (2 * sigma**2))

        def renorm(y1, x1):
            return y1 / torch.trapz(y1, x1, dim=-1).unsqueeze(-1)

        z = pdf(z, x, renorm=renorm)
        z = cdf(z, x, dim=-1)
        z_true = 0.5 * (1 + torch.erf((x - mu) / (sigma * np.sqrt(2))))
        torch.testing.assert_close(
            z, z_true, rtol=self.c.atol, atol=self.c.rtol
        )
