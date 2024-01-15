import torch
import torch.nn as nn


class W2Loss(nn.Module):
    def __init__(self, R):
        super().__init__()
        self.R = R

    def forward(self, f, g):
        # Apply the transformation R
        f_tilde = self.R(f)
        g_tilde = self.R(g)

        # Sort the distributions for inverse CDF computation
        sorted_f, _ = torch.sort(f_tilde, dim=1)
        sorted_g, _ = torch.sort(g_tilde, dim=1)

        # Compute the cumulative sums to approximate CDFs
        cum_f = torch.cumsum(sorted_f, dim=1)
        cum_g = torch.cumsum(sorted_g, dim=1)

        # Compute the inverse CDFs
        inv_cdf_f = torch.linspace(
            0, 1, steps=f.shape[1], device=f.device
        ).expand_as(cum_f)
        inv_cdf_g = torch.linspace(
            0, 1, steps=g.shape[1], device=g.device
        ).expand_as(cum_g)

        # Compute the W2 distance using the inverse CDFs
        w2_distance = torch.sqrt(torch.sum((inv_cdf_f - inv_cdf_g) ** 2, dim=1))

        return torch.mean(w2_distance)


def str_to_renorm(key):
    def abs_renorm(y):
        return torch.abs(y) / torch.sum(torch.abs(y), dim=1, keepdim=True)

    def square_renorm(y):
        return y**2 / torch.sum(y**2, dim=1, keepdim=True)

    options = {'abs': abs_renorm, 'square': square_renorm}
    return options[key]
