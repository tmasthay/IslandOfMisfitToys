import torch
import torch.nn as nn
from torch.nn.functional import mse_loss


class TikhonovLoss(nn.Module):
    def __init__(self, *, weights, alpha=1.0):
        """
        model_param: The single 600x250 parameter tensor of the model
        alpha: Regularization coefficient
        """
        super(TikhonovLoss, self).__init__()
        self.weights = weights
        self.alpha = alpha

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
        # Mean squared error
        least_squares = mse_loss(pred, target)

        # Gradient penalty
        grad_penalty = self.compute_gradient_penalty(self.weights)

        # Total loss
        total_loss = least_squares + self.alpha * grad_penalty
        return total_loss
