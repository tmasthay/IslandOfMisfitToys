import inspect

import torch
from mh.core import draise

from misfit_toys.utils import taper


def direct(*, scale=1.0):
    def helper(self):
        self.out = self.prop(None)[-1]

        # self.loss = scale * self.loss_fn(self.out, obs_data_filt)
        self.loss = scale * self.loss_fn(self.out)
        self.loss.backward()

    return helper
