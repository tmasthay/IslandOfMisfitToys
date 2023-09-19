from ...utils import auto_path, get_pydict
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
        ]=None
    ):
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
            
        self.obs_data = get(obs_data, 'obs_data')
        self.src_amp_y = get(src_amp_y, 'src_amp_y')
        self.src_amp_x = get(src_amp_x, 'src_amp_x')
        self.src_loc_y = get(src_loc_y, 'src_loc_y')
        self.rec_loc_y = get(rec_loc_y, 'rec_loc_y')
        self.vp = get(vp_init, 'vp_init')
        self.vs = get(vs_init, 'vs_init')
        self.rho = get(rho_init, 'rho_init')
        self.vp_true = get(vp_true, 'vp_true')
        self.vs_true = get(vs_true, 'vs_true')
        self.rho_true = get(rho_true, 'rho_true')
        self.src_amp_y_true = get(src_amp_y_true, 'src_amp_y_true')
        self.src_amp_x_true = get(src_amp_x_true, 'src_amp_x_true')

        self.metadata = get_pydict(path)

        self.set_meta_fields()
    
    def set_meta_fields(self):
        self.custom = {}
        for k,v in self.metadata.items():
            if( k in self.__slots__ ):
                setattr(self, k, v)
            else:
                self.custom[k] = v

