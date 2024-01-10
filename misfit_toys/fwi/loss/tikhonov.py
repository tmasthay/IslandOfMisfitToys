from typing import Callable
import torch
import torch.nn as nn
from torch.nn.functional import mse_loss
from dataclasses import dataclass


class TikhonovLoss(nn.Module):
    weights: torch.Tensor
    alpha: Callable[[float, float], float]
    max_iters: int

    def __init__(self, weights, *, alpha, max_iters=100):
        super().__init__()
        self.weights = weights
        self.alpha = alpha
        self.max_iters = max_iters
        self.iter = 0
        self.build_status = False
        self.status = "uninitialized"

    def compute_gradient_penalty(self, param):
        """
        Compute the gradient penalty for the parameter tensor
        """
        grad_x = torch.diff(param.p, dim=0)  # Gradient along x-axis (rows)
        grad_y = torch.diff(param.p, dim=1)  # Gradient along y-axis (columns)

        # Compute the norm of the gradient (you might choose L1, L2, etc.)
        penalty = grad_x.norm() + grad_y.norm()
        return penalty

    def forward(self, pred, target):
        """
        model_output: output from the model
        target: ground truth or target data
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
