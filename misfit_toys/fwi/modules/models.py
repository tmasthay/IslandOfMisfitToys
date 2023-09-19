from .seismic_data import SeismicData
from ...utils import DotDict

import torch
import deepwave as dw
from deepwave.common import vpvsrho_to_lambmubuoyancy as get_lame

from abc import abstractmethod
from typing import Annotated as Ant, Optional as Opt, Callable as Call


class Param(torch.nn.Module):
    model: Ant[ torch.nn.Parameter, 'Torch parameter' ] 
    custom: Ant[ DotDict, 'Custom parameters' ]

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
            maxv=maxv
        )

    def forward(self):
        minv = self.custom.minv
        maxv = self.custom.maxv
        return torch.sigmoid(self.p) * (maxv - minv) + minv
    
class Prop(torch.nn.Module):
    def __init__(
        self,
        *,
        data: Ant[SeismicData, "Seismic data tensors+metadata"],
        vp_prmzt: Ant[Call[[torch.Tensor], Param], "Parameterized vp"] \
            =Param.delay_init(requires_grad=True),
        vs_prmzt: Ant[Call[[torch.Tensor], Param], "Parameterized vs"] \
            =Param.delay_init(requires_grad=False),
        rho_prmzt: Ant[Call[[torch.Tensor], Param], "Parameterized rho"] \
            =Param.delay_init(requires_grad=False),
        src_amp_y_prmzt: Ant[
            Call[[torch.Tensor], Param], 
            "Parameterized src amp y"
        ]=Param.delay_init(requires_grad=False),
        src_amp_x_prmzt: Ant[
            Call[[torch.Tensor], Param], 
            "Parameterized src amp x"
        ]=Param.delay_init(requires_grad=False)
    ):
        self.data = data
        
        def reset(field, prmzt):
            #Note that you could easily set
            # setattr(self, field, ...) and then delete the plain tensor
            # consider this refactor if things break
            if( getattr(self.data, field) is not None ):
                setattr(self.data, field, prmzt(getattr(self.data, field)))

        reset('vp', vp_prmzt)
        reset('vs', vs_prmzt)
        reset('rho', rho_prmzt)
        reset('src_amp_y', src_amp_y_prmzt)
        reset('src_amp_x', src_amp_x_prmzt)

    def get_model(self):
        if( self.data.vp is None ):
            raise ValueError('vp must be set!')
        if( self.data.vs is None ):
            self.model = 'acoustic'
        elif( self.data.rho is None ):
            raise ValueError('Either rho and vs must be set, or neither!')
        else:
            self.model = 'elastic'
    
    def forward(self, **kw):
        if( self.model == 'acoustic' ):
            return dw.scalar(
                self.data.vp(),
                self.data.dy,
                self.data.dt,
                source_amplitudes=self.data.src_amp_y(),
                source_locations=self.data.src_loc,
                receiver_locations=self.data.rec_loc,
                **kw
            )[-1]
        else:
            return dw.elastic(
                *get_lame(self.data.vp(), self.data.vs(), self.data.rho()),
                self.data.dy,
                self.data.dt,
                source_amplitudes_y=self.data.src_amp_y(),
                source_locations_y=self.data.src_loc_y,
                receiver_locations_y=self.data.rec_loc_y,
                **kw
            )

# class Prop(torch.nn.Module):
#     def __init__(self, model, dx, dt, freq):
#         super().__init__()
#         self.model = model
#         self.dx = dx
#         self.dt = dt
#         self.freq = freq

#     def forward(self, source_amplitudes, source_locations,
#                 receiver_locations):
#         v = self.model()
#         return dw.scalar(
#             v, self.dx, self.dt,
#             source_amplitudes=source_amplitudes,
#             source_locations=source_locations,
#             receiver_locations=receiver_locations,
#             max_vel=2500,
#             pml_freq=self.freq,
#             time_pad_frac=0.2,
#         )