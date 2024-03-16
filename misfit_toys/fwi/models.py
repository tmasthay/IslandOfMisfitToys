"""
This module contains classes for representing parameters used in a neural network model.

Classes:
    Param: Represents a parameter used in a neural network model.
    ParamConstrained: Represents a constrained parameter used in a neural network model.
"""

from abc import abstractmethod
from dataclasses import dataclass
from typing import Annotated as Ant
from typing import Callable as Call
from typing import Optional as Opt

import deepwave as dw
import torch
from deepwave.common import vpvsrho_to_lambmubuoyancy as get_lame
from mh.core import DotDict


class Param(torch.nn.Module):
    """
    Represents a parameter used in a neural network model.

    Args:
        p (torch.Tensor): The parameter tensor.
        requires_grad (bool, optional): Whether the parameter requires gradient computation. Defaults to False.
        **kw: Additional custom attributes for the parameter.

    Attributes:
        p (torch.nn.Parameter): The parameter tensor wrapped as a `torch.nn.Parameter`.
        custom (DotDict): Additional custom attributes for the parameter.

    Methods:
        forward(): Returns the parameter tensor.

        delay_init(**kw): A class method that returns a lambda function for delayed initialization of the parameter.

    """

    def __init__(self, *, p, requires_grad=False, **kw):
        super().__init__()
        self.p = torch.nn.Parameter(p, requires_grad=requires_grad)
        self.custom = DotDict(kw)

    def forward(self):
        return self.p

    @classmethod
    def delay_init(cls, **kw):
        return lambda p: cls(p=p, **kw)


class ParamConstrained(Param):
    """
    Represents a constrained parameter used in a neural network model.

    Args:
        p (torch.Tensor): The parameter tensor.
        minv (float): The minimum value of the parameter.
        maxv (float): The maximum value of the parameter.
        requires_grad (bool, optional): Whether the parameter requires gradient computation. Defaults to False.

    Attributes:
        p (torch.nn.Parameter): The parameter tensor wrapped as a `torch.nn.Parameter`.
        custom (DotDict): Additional custom attributes for the parameter, including minv and maxv.

    Methods:
        forward(): Returns the constrained parameter tensor.

    """

    def __init__(self, *, p, minv, maxv, requires_grad=False):
        super().__init__(
            p=torch.logit((p - minv) / (maxv - minv)),
            requires_grad=requires_grad,
            minv=minv,
            maxv=maxv,
        )

    def forward(self):
        minv = self.custom.minv
        maxv = self.custom.maxv
        return torch.sigmoid(self.p) * (maxv - minv) + minv
