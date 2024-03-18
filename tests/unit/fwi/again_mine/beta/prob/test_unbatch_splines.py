import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from misfit_toys.beta import unbatch_splines


@given(
    frequency=st.floats(
        min_value=0.1, max_value=3.0, exclude_min=True, exclude_max=True
    )
)
@settings(max_examples=5)
def test_sine_wave_spline_with_random_frequency(frequency):
    # Define sine wave parameters with hypothesis-generated frequency
    x = torch.linspace(0, 2 * np.pi, steps=100)  # X values
    y = torch.sin(
        frequency * x
    )  # Y values using torch.sin() and random frequency

    # Create spline objects
    splines = unbatch_splines(x, y)

    # Define test points
    x_test = torch.linspace(0, 2 * np.pi, steps=1000)
    x_test = x_test[20:-20]
    y_true = torch.sin(frequency * x_test)
    y_deriv_true = frequency * torch.cos(frequency * x_test)

    for i, spline in enumerate(splines):
        # Evaluate spline and its derivative
        y_pred = spline.evaluate(x_test).squeeze()
        y_deriv_pred = spline.derivative(x_test).squeeze()

        # Assert the spline approximation is close to the true values
        torch.testing.assert_close(y_pred, y_true, atol=1e-2, rtol=1e-2)

        # Assert the spline derivative is close to the analytical derivative
        torch.testing.assert_close(
            y_deriv_pred, y_deriv_true, atol=1.0e-2, rtol=1.0e-2
        )
