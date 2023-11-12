import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


class Renorm:
    @staticmethod
    def square(x):
        u = x**2
        return u / u.sum()

    @staticmethod
    def shift(x):
        u = x - x.min()
        return u / u.sum()

    # @staticmethod
    # def exp(beta):
    #     def helper(x):
    #         u = torch.exp(beta * x)
    #         return u / u.sum()

    #     return helper
    @staticmethod
    def exp(x):
        u = torch.exp(0.1 * x)
        return u / u.sum()

    @staticmethod
    def get_options():
        return {
            k: v.__func__
            for k, v in Renorm.__dict__.items()
            if isinstance(v, staticmethod)
        }

    @staticmethod
    def choose(s, *args, **kw):
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
    def __init__(self, renorm_func, eps=1.0):
        super().__init__()
        self.renorm_func = renorm_func
        self.eps = eps

    def prep_data(self, y):
        y = self.eps + self.renorm_func(y)
        y = torch.cumulative_trapezoid(y, dim=-1)
        return y / y[..., -1].unsqueeze(-1)

    def forward(self, y_pred, y_true):
        y_pred = self.prep_data(y_pred)

        # Note that this is not necessary...should be a preprocessing step
        y_true = self.prep_data(y_true)
        abs_diff = torch.abs(y_pred - y_true)
        loss = torch.trapz(abs_diff, dim=-1).mean()
        return loss


class L2(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        loss = torch.mean((y_pred - y_true) ** 2)
        return loss


# This is incorrect implementation of W2
class W2(torch.nn.Module):
    def __init__(self, renorm_func, eps=0.0):
        super().__init__()
        self.renorm_func = renorm_func
        self.eps = eps

    def prep_data(self, y):
        y = self.eps + self.renorm_func(y)

    def forward(self, y_pred, y_true):
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
    def __init__(self, delta=0.5):
        super().__init__()
        self.delta = delta

    def forward(self, y_pred, y_true):
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
    def __init__(self, delta):
        super(HuberLoss, self).__init__()
        self.delta = torch.tensor(delta)

    def forward(self, input, target):
        error = input - target
        abs_error = torch.abs(error)
        quadratic = torch.minimum(abs_error, self.delta)
        linear = abs_error - quadratic
        loss = 0.5 * quadratic**2 + self.delta * linear
        return (
            loss.mean()
        )  # Assuming you want to average the loss over all elements


class Hybrid_norm(torch.nn.Module):
    def __init__(self, delta=10.0):
        super(Hybrid_norm, self).__init__()
        self.delta = delta

    def forward(self, y_pred, y_true):
        # Calculate the difference between the two seismic images
        r = y_pred - y_true

        # Apply the norm f(r) = sqrt(1 + (r/delta)^2) - 1
        h = torch.sqrt(1 + (r / self.delta) ** 2) - 1
        loss = torch.sum(h)

        return loss


class Tikhonov(torch.nn.Module):
    def __init__(self, lmbda, velocity, R):
        self.lmbda = lmbda
        self.velocity = velocity
        self.R = R


class GSOT(torch.nn.Module):
    def __init__(self, eta=0.003):
        super(GSOT, self).__init__()
        self.eta = eta

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
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
