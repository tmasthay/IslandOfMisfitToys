from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
from mh.core import DotDict
from torch.nn.functional import mse_loss


class TikhonovLoss(nn.Module):
    """
    Tikhonov regularization loss module.

    Attributes:
        weights (torch.Tensor): The weights tensor used for the regularization term.
        alpha (Callable[[float, float], float]): A callable function that takes the current iteration and maximum iterations and returns the regularization strength.
        max_iters (int): The maximum number of iterations.
        iter (int): The current iteration.
        build_status (bool): Whether the status is being built (unused/deprecated).
        status (str): The current status message (unused/deprecated).
    """

    def __init__(
        self,
        weights: torch.Tensor,
        *,
        alpha: Callable[[float, float], float],
        max_iters: int = 100,
    ):
        """
        Initializes the TikhonovLoss instance.

        Args:
            weights (torch.Tensor): The weights tensor used for the regularization term.
            alpha (Callable[[float, float], float]): A callable function that takes the current iteration and maximum iterations and returns the regularization strength.
            max_iters (int): The maximum number of iterations.
        """
        super().__init__()
        self.weights = weights
        self.alpha = alpha
        self.max_iters = max_iters
        self.iter = 0
        self.build_status = False
        self.status = "uninitialized"

    def compute_gradient_penalty(self, param):
        """
        Compute the gradient penalty for the parameter tensor.

        Args:
            param (torch.Tensor): The parameter tensor for which the gradient penalty is computed.

        Returns:
            torch.Tensor: The computed gradient penalty.
        """
        grad_x = torch.diff(param.p, dim=0)  # Gradient along x-axis (rows)
        grad_y = torch.diff(param.p, dim=1)  # Gradient along y-axis (columns)

        # Compute the norm of the gradient (you might choose L1, L2, etc.)
        penalty = grad_x.norm() + grad_y.norm()
        return penalty

    def forward(self, pred, target):
        """
        Compute the total loss including the least squares and Tikhonov regularization term.

        Args:
            pred (torch.Tensor): The model output predictions.
            target (torch.Tensor): The ground truth or target data.

        Returns:
            torch.Tensor: The total computed loss.
        """
        least_squares = mse_loss(pred, target)
        grad_penalty = self.compute_gradient_penalty(self.weights)
        reg_strength = self.alpha(self.iter, self.max_iters)

        total_loss = (
            least_squares + self.alpha(self.iter, self.max_iters) * grad_penalty
        )

        if self.build_status:
            self.status = (
                f"iter={self.iter}, loss={total_loss:.2e},"
                f" mse={least_squares:.2e},"
                f" tik={reg_strength * grad_penalty:.2e},"
                f" reg_strength={reg_strength:.2e}"
            )

            self.iter += 1

        return total_loss


def lin_reg_drop(
    *, weights, max_iters, scale, _min
) -> Callable[[int, int], float]:
    """
    Creates a dictionary with weights, alpha function for linear regularization drop,
    and maximum iterations.

    Args:
        weights (torch.Tensor): The weights tensor used for the regularization term.
        max_iters (int): The maximum number of iterations.
        scale (float): The scaling factor for the regularization strength.
        _min (float): The minimum regularization strength.

    Returns:
        DotDict: A dictionary containing weights, alpha function, and maximum iterations.
    """

    def reg_strength(iters, max_iters):
        return max(scale * (1 - iters / max_iters), _min)

    kw = DotDict(
        {'weights': weights, 'alpha': reg_strength, 'max_iters': max_iters}
    )

    return kw


def lin_reg_drop_legacy2(
    c: DotDict, *, scale, _min
) -> Callable[[int, int], float]:
    """
    Creates a dictionary with legacy configuration and alpha function for linear
    regularization drop.

    Args:
        c (DotDict): A configuration dictionary.
        scale (float): The scaling factor for the regularization strength.
        _min (float): The minimum regularization strength.

    Returns:
        tuple: A tuple containing an empty list and a dictionary with weights, alpha function,
        and maximum iterations.
    """

    def reg_strength(iters, max_iters):
        return max(scale * (1 - iters / max_iters), _min)

    kw = DotDict(
        {
            'weights': c.runtime.prop.module.vp,
            'alpha': reg_strength,
            'max_iters': c.train.max_iters,
        }
    )

    return [], kw


def lin_reg_drop_legacy(
    c: DotDict, *, scale, _min
) -> Callable[[int, int], float]:
    """
    Creates a dictionary with legacy configuration and alpha function for linear
    regularization drop, with validation for chosen loss type.

    Args:
        c (DotDict): A configuration dictionary.
        scale (float): The scaling factor for the regularization strength.
        _min (float): The minimum regularization strength.

    Returns:
        tuple: A tuple containing an empty list and a dictionary with weights, alpha function,
        and maximum iterations.

    Raises:
        ValueError: If the chosen loss type or regularization method is not 'tik' or 'lin_reg_drop'.
    """
    if c.train.loss.chosen.lower() != 'tik':
        raise ValueError(
            f"c.loss.chosen.lower() = {c.loss.chosen.lower()} != 'tik'"
        )
    if c.train.loss.tik.chosen.lower() != 'lin_reg_drop':
        raise ValueError(
            f"c.loss.tik.chosen.lower() = {c.loss.tik.chosen.lower()}"
            " != 'lin_reg_drop'"
        )

    def reg_strength(iters, max_iters):
        return max(scale * (1 - iters / max_iters), _min)

    kw = DotDict(
        {
            'weights': c.prop.module.vp,
            'alpha': reg_strength,
            'max_iters': c.train.max_iters,
        }
    )

    return [], kw


def lin_reg_tmp(c: DotDict) -> Callable[[int, int], float]:
    """
    Creates an alpha function for linear regularization drop based on temporary
    configuration.

    Args:
        c (DotDict): A configuration dictionary.

    Returns:
        Callable[[int, int], float]: A function that computes the regularization strength based on the iteration and maximum iterations.
    """

    def reg_strength(iters, max_iters):
        return max(
            c.train.loss.tik.lin_reg_drop.kw.scale * (1 - iters / max_iters),
            c.train.loss.tik.lin_reg_drop.kw._min,
        )

    return reg_strength
