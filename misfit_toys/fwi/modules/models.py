import torch
import deepwave as dw
from abc import abstractmethod
from ...utils import DotDict
from typing import Annotated as Ant, Optional as Opt

class ParamAbstract(torch.nn.Module):
    model: Ant[ torch.nn.Parameter, 'Torch parameter' ] 
    custom: Ant[ DotDict, 'Custom parameters' ]

    def __init__(self, *, p, requires_grad=False, **kw):
        super().__init__()
        self.p = torch.nn.Parameter(p, requires_grad=requires_grad)
        self.custom = DotDict(kw)

    @abstractmethod
    def forward(self):
        raise NotImplementedError
    
class Param(ParamAbstract):
    def __init__(self, *, p, requires_grad=False):
        super().__init__(p=p, requires_grad=requires_grad)

    def forward(self):
        return self.p
    
class ParamConstrained(ParamAbstract):
    def __init__(self, *, p, minv, maxv, requires_grad=False):
        super().__init__(
            p=torch.logit((p - minv) / (maxv - minv)),
            requires_grad=requires_grad, 
            minv=minv, 
            maxv=maxv
        )

    def forward(self):
        minv = self.custom.minv
        maxv = self.custom.maxv
        return torch.sigmoid(self.p) * (maxv - minv) + minv

class Prop(torch.nn.Module):
    def __init__(self, model, dx, dt, freq):
        super().__init__()
        self.model = model
        self.dx = dx
        self.dt = dt
        self.freq = freq

    def forward(self, source_amplitudes, source_locations,
                receiver_locations):
        v = self.model()
        return dw.scalar(
            v, self.dx, self.dt,
            source_amplitudes=source_amplitudes,
            source_locations=source_locations,
            receiver_locations=receiver_locations,
            max_vel=2500,
            pml_freq=self.freq,
            time_pad_frac=0.2,
        )