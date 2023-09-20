from ...utils import auto_path, get_pydict, SlotMeta, DotDict
from ...data.dataset import *
from .models import Param, ParamConstrained

from deepwave.common import vpvsrho_to_lambmubuoyancy as get_lame

from torchaudio.functional import biquad
from typing import Annotated as Ant, Optional as Opt, Union, Callable as Call
from scipy.ndimage import gaussian_filter
from dataclasses import dataclass, field
import json


class SeismicProp(torch.nn.Module, metaclass=SlotMeta):
    obs_data: Ant[torch.Tensor, 'Observed data']
    src_amp_y: Ant[torch.Tensor, 'Source amplitude, y component']
    src_amp_x: Opt[Ant[torch.Tensor, 'Source amplitude, x component']]
    src_loc_y: Ant[torch.Tensor, 'Source locations']
    rec_loc_y: Ant[torch.Tensor, 'Receiver locations']
    vp: Ant[torch.Tensor, 'Initial P velocity model']
    vs: Opt[Ant[torch.Tensor, 'Initial S velocity model']]
    rho: Opt[Ant[torch.Tensor, 'Initial density model']]
    vp_true: Opt[Ant[torch.Tensor, 'True P velocity model']]
    vs_true: Opt[Ant[torch.Tensor, 'True S velocity model']]
    rho_true: Opt[Ant[torch.Tensor, 'True density model']]
    src_amp_y_true: Opt[Ant[torch.Tensor, 'True source amplitude, y component']]
    src_amp_x_true: Opt[Ant[torch.Tensor, 'True source amplitude, x component']]
    nx: Ant[int, 'Number of x grid points']
    ny: Ant[int, 'Number of y grid points']
    nt: Ant[int, 'Number of time steps']
    dx: Ant[float, 'Grid spacing in x']
    dy: Ant[float, 'Grid spacing in y']
    dt: Ant[float, 'Time step']
    n_shots: Ant[int, 'Number of shots']
    src_per_shot: Ant[int, 'Number of sources per shot']
    rec_per_shot: Ant[int, 'Number of receivers per shot']
    freq: Ant[float, 'Source frequency']
    metadata: Ant[dict, 'Metadata']
    custom: Ant[dict, 'Custom data']
    
    @auto_path(make_dir=False)
    def __init__(
        self,
        *,
        path: Ant[str, 'Path to data'],
        obs_data: Ant[Union[str, torch.Tensor] , 'obs_data']=None,
        src_amp_y: Ant[Union[str, torch.Tensor], 'Source amp. y']=None,
        src_loc_y: Ant[Union[str, torch.Tensor], 'Source locations']=None,
        rec_loc_y: Ant[Union[str, torch.Tensor], 'Receiver locations']=None,
        vp_init: Ant[
            Union[str, torch.Tensor], 
            'Initial P velocity model'
        ]=None,
        src_amp_x: Opt[Ant[Union[str, torch.Tensor], 'Source amp. x']]=None,
        vs_init: Opt[Ant[Union[str, torch.Tensor], 'Init S vel']]=None,
        rho_init: Opt[Ant[Union[str, torch.Tensor], 'Init density']]=None,
        vp_true: Opt[Ant[Union[str, torch.Tensor], 'True P vel']]=None,
        vs_true: Opt[Ant[Union[str, torch.Tensor], 'True S vel']]=None,
        rho_true: Opt[Ant[Union[str, torch.Tensor], 'True density ']]=None,
        src_amp_y_true: Opt[
            Ant[Union[str, torch.Tensor], 'True source amp. y']
        ]=None,
        src_amp_x_true: Opt[
            Ant[Union[str, torch.Tensor], 'True source amp. x']
        ]=None,
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
        super().__init__()
        def get(filename, default=None):
            if( isinstance(filename, torch.Tensor) ):
                return filename
            elif( filename is not None ):
                return get_data2(field=filename, path=path)
            elif( filename is None and default is not None ):
                if( os.path.exists(f'{path}/{default}.pt') ):
                    return get_data2(field=default, path=path)
                else:
                    return None
            else:
                return None
        
        def get_prmzt(filename, default=None, *, prmzt):
            tmp = get(filename, default=default)
            if( tmp is None ):
                return None
            else:
                return prmzt(tmp)
        
        self.vp = get_prmzt(vp_init, 'vp_init', prmzt=vp_prmzt)
        self.vs = get_prmzt(vs_init, 'vs_init', prmzt=vs_prmzt)
        self.rho = get_prmzt(rho_init, 'rho_init', prmzt=rho_prmzt)
        self.src_amp_y = get_prmzt(
            src_amp_y, 
            'src_amp_y', 
            prmzt=src_amp_y_prmzt
        )
        self.src_amp_x = get_prmzt(
            src_amp_x, 
            'src_amp_y', 
            prmzt=src_amp_x_prmzt
        )

        self.obs_data = get(obs_data, 'obs_data')
        self.src_loc_y = get(src_loc_y, 'src_loc_y')
        self.rec_loc_y = get(rec_loc_y, 'rec_loc_y')
        self.vp_true = get(vp_true, 'vp_true')
        self.vs_true = get(vs_true, 'vs_true')
        self.rho_true = get(rho_true, 'rho_true')
        self.src_amp_y_true = get(src_amp_y_true, 'src_amp_y_true')
        self.src_amp_x_true = get(src_amp_x_true, 'src_amp_x_true')

        self.model = 'acoustic' if self.vs is None else 'elastic'

        self.metadata = get_pydict(path)

        self.set_meta_fields()
    
    def set_meta_fields(self):
        custom_dict = {}
        for k,v in self.metadata.items():
            if( k in self.__slots__ ):
                setattr(self, k, v)
            else:
                custom_dict[k] = v
        self.custom = DotDict(custom_dict)

    def forward(self, **kw):
        if( self.model == 'acoustic' ):
            return dw.scalar(
                self.vp(),
                self.dy,
                self.dt,
                source_amplitudes=self.src_amp_y(),
                source_locations=self.src_loc_y,
                receiver_locations=self.rec_loc_y,
                **kw
            )[-1]
        else:
            return dw.elastic(
                *get_lame(self.vp(), self.vs(), self.rho()),
                self.dy,
                self.dt,
                source_amplitudes_y=self.src_amp_y(),
                source_locations_y=self.src_loc_y,
                receiver_locations_y=self.rec_loc_y,
                **kw
            )

