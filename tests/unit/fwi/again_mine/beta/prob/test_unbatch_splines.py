import numpy as np
import pytest
import torch
from mh.core import draise

from misfit_toys.beta import unbatch_splines


def test_sine_wave_spline():
    # Define the sine wave parameters
    frequency = 1.0  # Frequency of sine wave
    x = torch.linspace(0, 2 * np.pi, steps=100)  # X values
    y = torch.sin(frequency * x)  # Y values using torch.sin()

    # Create spline objects
    splines = unbatch_splines(x, y)

    # Define test points
    x_test = torch.linspace(0, 2 * np.pi, steps=100)
    y_true = torch.sin(frequency * x_test)
    y_deriv_true = frequency * torch.cos(frequency * x_test)

    for i, spline in enumerate(splines):
        # Evaluate spline and its derivative
        y_pred = spline.evaluate(x_test).squeeze()
        y_deriv_pred = spline.derivative(x_test).squeeze()

        # Assert the spline approximation is close to the true values
        torch.testing.assert_allclose(y_pred, y_true, atol=1e-2, rtol=1e-2)

        # Assert the spline derivative is close to the analytical derivative
        torch.testing.assert_allclose(
            y_deriv_pred, y_deriv_true, atol=1e-2, rtol=1e-2
        )
