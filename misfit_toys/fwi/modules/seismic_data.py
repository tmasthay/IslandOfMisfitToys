from ...utils import *
from torchaudio.functional import biquad
from typing import Annotated as Ant, Optional as Opt, Union
from scipy.ndimage import gaussian_filter
from dataclasses import dataclass, field
import json
from ...data.dataset import *

@dataclass(slots=True)
class SeismicData:
    obs_data: Ant[torch.Tensor, 'Observed data']
    src_amp_y: Ant[torch.Tensor, 'Source amplitude, y component']
    src_amp_x: Opt[Ant[torch.Tensor, 'Source amplitude, x component']]
    src_loc: Ant[torch.Tensor, 'Source locations']
    rec_loc: Ant[torch.Tensor, 'Receiver locations']
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
    custom: Ant[dict, 'Custom data']
    
    def __init__(
        self,
        *,
        obs_data: Ant[Union[str, torch.Tensor] , 'obs_data']='obs_data',
        src_amp_y: Ant[Union[str, torch.Tensor], 'Source amp. y']='src_amp_y',
        src_loc: Ant[Union[str, torch.Tensor], 'Source locations']='src_loc',
        rec_loc: Ant[Union[str, torch.Tensor], 'Receiver locations']='rec_loc',
        vp_init: Ant[
            Union[str, torch.Tensor], 
            'Initial P velocity model'
        ]='vp_init',
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
        path: Ant[str, 'Path to data']
    ):
        def get(filename):
            if( isinstance(filename, torch.Tensor) ):
                return filename
            elif( filename is not None ):
                return get_data2(field=filename, path=path)
            else:
                return None
            
        self.obs_data = get(obs_data)
        self.src_amp_y = get(src_amp_y)
        self.src_amp_x = get(src_amp_x)
        self.src_loc = get(src_loc)
        self.rec_loc = get(rec_loc)
        self.vp = get(vp_init)
        self.vs = get(vs_init)
        self.rho = get(rho_init)
        self.vp_true = get(vp_true)
        self.vs_true = get(vs_true)
        self.rho_true = get(rho_true)
        self.src_amp_y_true = get(src_amp_y_true)
        self.src_amp_x_true = get(src_amp_x_true)

        dynamic = ['ny', 'nx', 'nt', 'n_shots', 'src_per_shot', 'rec_per_shot']
        self.ny, self.nx = self.vp_init.shape
        self.nt = self.obs_data.shape[-1]
        self.n_shots = self.src_amp_y.shape[0]
        self.src_per_shot = self.src_amp_y.shape[1]
        self.rec_per_shot = self.rec_loc.shape[1]

        metadata = eval(open(f'{path}/metadata.json', 'r').read())
        self.custom = {}
        for k,v in metadata.items():
            if( k in self.__slots__ and k in dynamic ):
                if( getattr(self, k) != v ):
                    raise ValueError(
                        f"Metadata value for {k} does not match data"
                        f'self.{k} = {getattr(self, k)}' 
                        f' but metadata[{k}] = {v}'
                    )
            elif( k in self.__slots__ ):
                setattr(self, k, v)
            else:
                self.custom[k] = v

