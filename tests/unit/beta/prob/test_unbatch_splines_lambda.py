import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from misfit_toys.beta import unbatch_splines_lambda


@given(
    freq=st.floats(
        min_value=0.1, max_value=1.0, exclude_min=True, exclude_max=True
    )
)
@settings(max_examples=5)
def test_sine_wave_spline_with_random_frequency_lambda(sine_ref_data, freq):
    x, y, x_test, y_true, y_deriv_true, atol, rtol = sine_ref_data(freq)

    F = unbatch_splines_lambda(x, y)
    res = F(x_test).squeeze()
    res_deriv = F(x_test, deriv=True).squeeze()

    torch.testing.assert_close(res, y_true, atol=atol, rtol=rtol)
    torch.testing.assert_close(res_deriv, y_deriv_true, atol=atol, rtol=rtol)
