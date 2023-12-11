"""
This module defines custom loss functions for seismic image analysis.

Available Loss Functions:
- W1: Computes the W1 loss between predicted and true seismic images.
- L2: Computes the L2 loss between predicted and true seismic images.
- W2: Computes the W2 loss between predicted and true seismic images.
- HuberLegacy: Computes the Huber loss between predicted and true seismic images.
- HuberLoss: Computes the Huber loss between predicted and true seismic images.
- Hybrid_norm: Computes the hybrid norm loss between predicted and true seismic images.
- Tikhonov: Placeholder class for Tikhonov regularization loss.
- GSOT: Computes the GSOT loss between predicted and true seismic images.
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


class Renorm:
    """
    Class representing different renormalization methods.
    """

    @staticmethod
    def square(x):
        """Square renormalization method.

        Args:
            x: Input tensor.

        Returns:
            Tensor: Renormalized tensor.
        """
        u = x**2
        return u / u.sum()

    @staticmethod
    def shift(x):
        """Shift renormalization method.

        Args:
            x: Input tensor.

        Returns:
            Tensor: Renormalized tensor.
        """
        u = x - x.min()
        return u / u.sum()

    @staticmethod
    def exp(x):
        """Exponential renormalization method.

        Args:
            x: Input tensor.

        Returns:
            Tensor: Renormalized tensor.
        """
        u = torch.exp(0.1 * x)
        return u / u.sum()

    @staticmethod
    def get_options():
        """Get the available renormalization options.

        Returns:
            dict: A dictionary of available renormalization options.
        """
        return {
            k: v.__func__
            for k, v in Renorm.__dict__.items()
            if isinstance(v, staticmethod)
        }

    @staticmethod
    def choose(s, *args, **kw):
        """Choose a renormalization method.

        Args:
            s: Renormalization method name.
            *args: Additional positional arguments.
            **kw: Additional keyword arguments.

        Returns:
            function: The chosen renormalization method.

        Raises:
            ValueError: If the specified renormalization method is not recognized.
        """
        options = Renorm.get_options()
        if s not in options.keys():
            raise ValueError(
                f"Renorm option {s} not recognized...choose from"
                f" {options.keys()}"
            )
        if len(args) == 0 and len(kw.keys()) == 0:
            return options[s]
        else:
            return options[s](*args, **kw)


class W1(torch.nn.Module):
    """
    W1 loss function implementation.

    Args:
        renorm_func (callable): Function to renormalize the input data.
        eps (float, optional): Epsilon value for numerical stability. Defaults to 1.0.
    """

    def __init__(self, renorm_func, eps=1.0):
        super().__init__()
        self.renorm_func = renorm_func
        self.eps = eps

    def prep_data(self, y):
        """
        Preprocesses the input data.

        Args:
            y (torch.Tensor): Input data.

        Returns:
            torch.Tensor: Preprocessed data.
        """
        y = self.eps + self.renorm_func(y)
        y = torch.cumulative_trapezoid(y, dim=-1)
        return y / y[..., -1].unsqueeze(-1)

    def forward(self, y_pred, y_true):
        """
        Forward pass of the W1 loss function.

        Args:
            y_pred (torch.Tensor): Predicted values.
            y_true (torch.Tensor): True values.

        Returns:
            torch.Tensor: Loss value.
        """
        y_pred = self.prep_data(y_pred)

        # Note that this is not necessary...should be a preprocessing step
        y_true = self.prep_data(y_true)
        abs_diff = torch.abs(y_pred - y_true)
        loss = torch.trapz(abs_diff, dim=-1).mean()
        return loss


class L2(torch.nn.Module):
    """
    Calculates the L2 loss between predicted and true values.
    """

    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        """
        Calculates the L2 loss between predicted and true values.

        Args:
            y_pred (torch.Tensor): Predicted values.
            y_true (torch.Tensor): True values.

        Returns:
            torch.Tensor: L2 loss.
        """
        loss = torch.mean((y_pred - y_true) ** 2)
        return loss


# This is incorrect implementation of W2
class W2(torch.nn.Module):
    """
    W2 custom loss function.

    Args:
        renorm_func (callable): Function to renormalize the input data.
        eps (float, optional): Small value added to the input data to avoid division by zero. Defaults to 0.0.
    """

    def __init__(self, renorm_func, eps=0.0):
        super().__init__()
        self.renorm_func = renorm_func
        self.eps = eps

    def prep_data(self, y):
        """
        Preprocesses the input data.

        Args:
            y (torch.Tensor): Input data.

        Returns:
            torch.Tensor: Preprocessed data.
        """
        y = self.eps + self.renorm_func(y)

    def forward(self, y_pred, y_true):
        """
        Forward pass of the W2 loss function.

        Args:
            y_pred (torch.Tensor): Predicted values.
            y_true (torch.Tensor): True values.

        Returns:
            torch.Tensor: Loss value.
        """
        # Square the values
        y_pred = self.renorm_func(y_pred)
        y_true = self.renorm_func(y_true)

        # Calculate trapezoidal numerical integration
        y_pred_integral = torch.trapz(y_pred, dim=-1)
        y_true_integral = torch.trapz(y_true, dim=-1)

        # Divide by integral
        y_pred = y_pred / y_pred_integral
        y_true = y_true / y_true_integral

        # Calculate cumulative sum
        y_pred = torch.cumsum(y_pred, dim=-1)
        y_true = torch.cumsum(y_true, dim=-1)

        # Calculate squared difference and integrate
        squared_diff = (y_pred - y_true) ** 2
        loss = torch.trapz(squared_diff, dim=-1)

        return loss


class HuberLegacy(torch.nn.Module):
    """
    Huber loss function for regression tasks.

    Args:
        delta (float): The threshold value for the Huber loss function. Defaults to 0.5.

    Returns:
        torch.Tensor: The calculated loss value.

    """

    def __init__(self, delta=0.5):
        super().__init__()
        self.delta = delta

    def forward(self, y_pred, y_true):
        """
        Forward pass of the Huber loss function.

        Args:
            y_pred (torch.Tensor): The predicted values.
            y_true (torch.Tensor): The true values.

        Returns:
            torch.Tensor: The calculated loss value.

        """
        # Calculate the absolute difference between the two seismic images
        diff = torch.abs(y_pred - y_true)

        # Apply the Huber loss function to each difference value
        square_h = 0.5 * (y_pred - y_true) ** 2
        linear_h = (
            self.delta * torch.abs(y_pred - y_true) - 0.5 * self.delta**2
        )
        h = torch.where(diff <= self.delta, square_h, linear_h)

        # Average over all elements in the images
        loss = torch.sum(h) / torch.numel(y_pred)

        return loss


class HuberLoss(torch.nn.Module):
    """
    Huber loss function for regression tasks.

    Args:
        delta (float): The threshold value for the loss function.

    Returns:
        torch.Tensor: The computed loss value.
    """

    def __init__(self, delta):
        super(HuberLoss, self).__init__()
        self.delta = torch.tensor(delta)

    def forward(self, input, target):
        """
        Compute the Huber loss between the input and target tensors.

        Args:
            input (torch.Tensor): The predicted values.
            target (torch.Tensor): The target values.

        Returns:
            torch.Tensor: The computed loss value.
        """
        error = input - target
        abs_error = torch.abs(error)
        quadratic = torch.minimum(abs_error, self.delta)
        linear = abs_error - quadratic
        loss = 0.5 * quadratic**2 + self.delta * linear
        return (
            loss.mean()
        )  # Assuming you want to average the loss over all elements


class Hybrid_norm(torch.nn.Module):
    """
    Custom loss function that calculates the hybrid norm between predicted and true seismic images.

    Args:
        delta (float): The delta value used in the hybrid norm calculation.

    Returns:
        torch.Tensor: The hybrid norm loss value.
    """

    def __init__(self, delta=10.0):
        super(Hybrid_norm, self).__init__()
        self.delta = delta

    def forward(self, y_pred, y_true):
        """
        Forward pass of the hybrid norm loss function.

        Args:
            y_pred (torch.Tensor): The predicted seismic image.
            y_true (torch.Tensor): The true seismic image.

        Returns:
            torch.Tensor: The hybrid norm loss value.
        """
        # Calculate the difference between the two seismic images
        r = y_pred - y_true

        # Apply the norm f(r) = sqrt(1 + (r/delta)^2) - 1
        h = torch.sqrt(1 + (r / self.delta) ** 2) - 1
        loss = torch.sum(h)

        return loss


class Tikhonov(torch.nn.Module):
    """
    Tikhonov regularization loss module.

    Args:
        lmbda (float): The regularization parameter.
        velocity (torch.Tensor): The velocity tensor.
        R (torch.Tensor): The regularization matrix.
    """

    def __init__(self, lmbda, velocity, R):
        self.lmbda = lmbda
        self.velocity = velocity
        self.R = R


class GSOT(torch.nn.Module):
    """
    GSOT (Graph Space Optimal Transport) views 1D data as a point cloud and
    defines a Gaussian mixture around each point. The loss function between two
    signals is the 2D optimal transport distance between the two point clouds.
    """

    def __init__(self, eta=0.003):
        super(GSOT, self).__init__()
        self.eta = eta

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        """
        Forward pass of the GSOT loss function.

        Args:
            y_pred (torch.Tensor): The predicted tensor.
            y_true (torch.Tensor): The true tensor.

        Returns:
            torch.Tensor: The calculated loss.
        """
        loss = torch.tensor(0, dtype=torch.float)
        for s in range(y_true.shape[0]):
            for r in range(y_true.shape[1]):
                nt = y_true.shape[-1]
                c = np.zeros([nt, nt])
                for i in range(nt):
                    for j in range(nt):
                        c[i, j] = (
                            self.eta * (i - j) ** 2
                            + (y_pred.detach()[s, r, i] - y_true[s, r, j]) ** 2
                        )
                row_ind, col_ind = linear_sum_assignment(c)
                y_sigma = y_true[s, r, col_ind]
                loss = (
                    loss
                    + (
                        self.eta
                        * torch.tensor(
                            (row_ind - col_ind) ** 2, device=y_pred.device
                        )
                        + (y_pred[s, r] - y_sigma) ** 2
                    ).sum()
                )
        return loss
