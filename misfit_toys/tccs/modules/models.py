from abc import abstractmethod
from dataclasses import dataclass
from typing import Annotated as Ant
from typing import Callable as Call
from typing import Optional as Opt

import deepwave as dw
import torch
from deepwave.common import vpvsrho_to_lambmubuoyancy as get_lame
from masthay_helpers import DotDict


class Param(torch.nn.Module):
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
