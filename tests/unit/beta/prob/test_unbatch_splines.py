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
def test_sine_wave_spline_with_random_frequency(sine_ref_data, freq):
    x, y, x_test, y_true, y_deriv_true, atol, rtol = sine_ref_data(freq)
    splines = unbatch_splines(x, y)
    for i, spline in enumerate(splines):
        y_pred = spline.evaluate(x_test).squeeze()
        y_deriv_pred = spline.derivative(x_test).squeeze()

        torch.testing.assert_close(y_pred, y_true, atol=atol, rtol=rtol)
        torch.testing.assert_close(
            y_deriv_pred, y_deriv_true, atol=atol, rtol=rtol
        )
