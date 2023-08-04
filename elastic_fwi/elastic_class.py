import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
import deepwave
from deepwave import elastic
from custom_losses import *
from deepwave_helpers import *
import numpy as np
import argparse
from random import randint
from warnings import warn
from typing import Annotated as Ant
from abc import ABC, abstractmethod

global device
global cmap
device = torch.device('cuda:0')

plt.rc('text', usetex=False)

class Data:
    # """
    # This class represents a seismic data object.

    # Attributes
    # ----------
    # devices : list
    #     List of devices for computation (CPUs, GPUs).

    # nx : int
    #     Number of horizontal dofs. 
    #     1 <= nx

    # ny : int
    #     Number of vertical dofs. 
    #     1 <= ny

    # nt : int
    #     Number of time steps. 
    #     1 <= nt

    # dx : float
    #     Grid size in the horizontal direction.
    #     dx > 0.0

    # dy : float
    #     Grid size in the vertical direction.
    #     dy > 0.0

    # dt : float
    #     Time step. 
    #     dt > 0.0

    # vp : torch.Tensor
    #     P-wave velocity. 
    #     vp >= 0.0

    # vs : torch.Tensor
    #     S-wave velocity. 
    #     0 <= vs <= vp

    # rho : torch.Tensor
    #     Density.
    #     rho >= 0.0

    # n_shots : int
    #     Number of shots. 
    #     1 <= n_shots

    # fst_src : int
    #     First source index (x-direction).
    #     1 <= fst_src <= nx - 1

    # src_depth : int
    #     Source depth index (y-direction).
    #     1 <= src_depth <= ny - 1

    # d_src : int
    #     Index delta of sources (x-direction).
    #     1 <= d_src <= nx - 2

    # n_src_per_shot : int
    #     Number of sources per shot.
    #     n_src_per_shot >= 1

    # src_loc : torch.Tensor
    #     Source locations.

    # fst_rec : int
    #     First receiver index (x-direction).
    #     1 <= fst_rec <= nx - 1

    # rec_depth : int
    #     Receiver depth index (y-direction).
    #     1 <= rec_depth <= ny - 1

    # d_rec : int
    #     Index delta of receivers (x-direction).
    #     1 <= d_rec <= nx - 2

    # n_rec_per_shot : int
    #     Number of receivers per shot.
    #     n_rec_per_shot >= 1

    # rec_loc : torch.Tensor
    #     Receiver locations.

    # freq : float
    #     Characteristic Ricker frequency. 
    #     freq > 0.0

    # wavelet : torch.Tensor
    #     Characteristic source time signature.

    # ofs : int
    #     Padding for source location landscape. 
    #     1 <= ofs <= min(nx - 1, ny - 1)
    # """
    __slots__ = [
        'devices',

        'nx', 
        'ny',
        'nt', 
        'dx', 
        'dy',
        'dt',

        'vp', 
        'vs', 
        'rho', 
         
        'n_shots', 

        'fst_src', 
        'src_depth', 
        'd_src',
        'src_loc', 
        'n_src_per_shot',

        'fst_rec', 
        'rec_depth', 
        'd_rec',
        'n_rec_per_shot', 
        'rec_loc', 

        'freq',  
        'wavelet',
        
        'ofs', 
    ]
    devices: Ant[list, 'List of devices']

    vp: Ant[torch.Tensor, 'P-wave velocity', '0.0']
    vs: Ant[torch.Tensor, 'S-wave velocity', '0.0', 'vp']
    rho: Ant[torch.Tensor, 'Density', '0.0<']

    n_shots: Ant[int, 'Number of shots']

    fst_src: Ant[int, 'First source index (x-dir)', '1', 'nx-1']
    src_depth: Ant[int, 'Source depth index (y-dir)', '1', 'ny-1']
    d_src: Ant[int, 'Index delta of sources (x-dir)', '1', 'nx-2']
    n_src_per_shot: Ant[int, 'Num sources per shot', '1']
    src_loc: Ant[torch.Tensor, 'Source locations']

    fst_rec: Ant[int, 'First receiver index (x-dir)', '1', 'nx-1']
    rec_depth: Ant[int, 'Rec. depth index (y-dir)', '1', 'ny-1']
    d_rec: Ant[int, 'Index delta of receivers (x-dir)', '1', 'nx-2']
    n_rec_per_shot: Ant[int, 'Num receivers per shot', '1']
    rec_loc: Ant[torch.Tensor, 'Receiver locations']

    nx: Ant[int, 'Number of horizontal dofs', '1']
    ny: Ant[int, 'Number of vertical dofs', '1']
    nt: Ant[int, 'Number of time steps', '1']
    dx: Ant[float, 'Grid size horizontal', '0.0<']
    dy: Ant[float, 'Grid size vertical', '0.0<']
    dt: Ant[float, 'Time step', '0.0<']

    freq: Ant[float, 'Characteristic Ricker frequency', '0.0<']
    wavelet: Ant[torch.Tensor, 'Characteristic source time signature']

    ofs: Ant[int, 'Padding for src_loc landscape', '1', 'min(nx-1,ny-1)']

    def __init__(self, **kw):
        self.devices = [torch.device('cuda:%d'%i) \
            for i in range(torch.cuda.device_count())] + [torch.device('cpu')]
        self.vp = read_tensor(kw['vp'], self.devices[0])
        self.vs = read_tensor(kw['vs'], self.devices[0])
        self.rho = read_tensor(kw['rho'], self.devices[0])

        assert len(self.vp.shape) == 2, 'vp dimension mismatch, ' \
            f'Expected 2 dimensions but got {len(self.vp.shape)} in the ' \
            f'shape of {self.vp.shape}'

        assert self.vp.shape == self.vs.shape \
            and self.vs.shape == self.rho.shape, \
            'vp,vs,rho dimension mismatch, ' \
                f'{self.vp.shape},{self.vs.shape},{self.rho.shape}'

        #get shot info
        self.n_shots = kw['n_shots']

        #get source info
        self.fst_src = kw['fst_src']
        self.n_src_per_shot = kw['n_src_per_shot']
        self.src_depth = kw['src_depth']
        self.d_src = kw['d_src']
        self.src_loc = kw['src_loc']
    
        #get receiver info
        self.fst_rec = kw['fst_rec']
        self.n_rec_per_shot = kw['n_rec_per_shot']
        self.rec_depth = kw['rec_depth']
        self.d_rec = kw['d_rec']
        self.rec_loc = kw['rec_loc']

        #get grid info
        self.ny, self.nx, self.nt = *self.vp.shape, kw['nt']
        self.dy, self.dx, self.dt = kw['dy'], kw['dx'], kw['dt']

        #get time signature info
        self.freq = kw['freq']
        self.wavelet = deepwave.wavelets.ricker(kw['freq'], 
            kw['nt'], 
            kw['dt'],
            kw['peak_time']
        )
    
        #get offset info for optimization landscape plots
        self.ofs = kw.get('ofs', 1)\

    def update(self, **kw):
        for k,v in kw.items():
            if( k in self.__slots__ ):
                setattr(self, k, v)
            else:
                warn(f'Attribute {k} does not exist...skipping ')

class DataGenerator(Data, ABC):

    # """
    # This class is an abstract base class for generating seismic data. 
    # It inherits from the Data class and extends it by adding a custom parameter 
    #     for user flexibility. 

    # Attributes
    # ----------
    # custom : dict
    #     Custom parameters for user flexibility.

    # Methods
    # -------
    # force(y, x, comp, **kw)
    #     Abstract method for force calculation.

    # forward(**kw)
    #     Abstract method for forward modeling.
    # """
    __slots__ = [
        'custom'
    ]
    custom: Ant[dict, 'Custom parameters for user flexibility']
    def __init__(self, **kw):
        """_summary_
        """
        super().__init__(**kw)
        self.custom = dict()
        new_keys = set(kw.keys()).difference(set(super().__slots__))
        for k in new_keys:
            self.custom[k] = kw[k]
            
    @abstractmethod
    def force(y,x,comp,**kw):
        pass

    @abstractmethod
    def forward(**kw):
        pass
