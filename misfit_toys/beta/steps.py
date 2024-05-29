"""
Centralized locations for _step attribute override of Training object in misfit_toys.examples.hydra.main.
"""

import inspect

import torch
from mh.core import draise

from misfit_toys.utils import taper


def direct(*, scale=1.0):
    """
    A decorator that simply directly compares the output to the observed data, without filtering.

    Args:
        scale (float, optional): The scale factor to apply to the loss function. Defaults to 1.0.

    Returns:
        callable: A helper function that performs the backward propagation.

    """

    def helper(self):
        self.out = self.prop(None)[-1]

        # self.loss = scale * self.loss_fn(self.out, obs_data_filt)
        self.loss = scale * self.loss_fn(self.out)
        self.loss.backward()

    return helper


# def direct_softplus(*, obs_data, scale=1.0):
#     transformed_data = torch.log(1 + torch.exp(obs_data * scale)) / scale

#     def helper(self):
#         self.out = self.prop(None)[-1]
#         out_transformed = torch.log(1 + torch.exp(self.out * scale)) / scale

#         # self.loss = scale * self.loss_fn(self.out, obs_data_filt)
#         self.loss = scale * self.loss_fn(out_transformed, transformed_data)
#         self.loss.backward()
