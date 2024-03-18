import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from misfit_toys.beta import unbatch_splines


@given(
    freq=st.floats(
        min_value=0.1, max_value=1.0, exclude_min=True, exclude_max=True
    )
)
@settings(max_examples=5)
def test_sine_wave_spline_with_random_frequency(cfg, adjust, freq):
    c = cfg.unit.beta.prob.unbatch_splines
    freq = adjust(freq, *c.freq)

    x = torch.linspace(c.x.left, c.x.right, c.x.num_ref)
    y = torch.sin(freq * x)

    splines = unbatch_splines(x, y)

    x_test = torch.linspace(c.x.left, c.x.right, c.x.num_ref * c.x.upsample)
    x_test = x_test[c.pad : -c.pad]
    y_true = torch.sin(freq * x_test)
    y_deriv_true = freq * torch.cos(freq * x_test)

    atol = c.get('atol', cfg.atol)
    rtol = c.get('rtol', cfg.rtol)
    for i, spline in enumerate(splines):
        y_pred = spline.evaluate(x_test).squeeze()
        y_deriv_pred = spline.derivative(x_test).squeeze()

        torch.testing.assert_close(y_pred, y_true, atol=atol, rtol=rtol)

        torch.testing.assert_close(
            y_deriv_pred, y_deriv_true, atol=atol, rtol=rtol
        )
