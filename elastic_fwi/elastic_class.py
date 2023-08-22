import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
import deepwave
from deepwave import elastic
import numpy as np
import argparse
from random import randint
from warnings import warn
from typing import Annotated as Ant
from abc import ABC, abstractmethod

from .custom_losses import *
from .deepwave_helpers import *

plt.rc('text', usetex=False)

class Data(metaclass=SlotMeta):
    """
    ***
    Class for parsing data input for FWI.
    ***
    """
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

    src_amplitudes: Ant[torch.Tensor, 'Source amplitudes']

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
        self.devices = get_all_devices()
        self.vp = read_tensor(kw['vp'], self.devices[0])
        self.vs = read_tensor(kw['vs'], self.devices[0])
        self.rho = read_tensor(kw['rho'], self.devices[0])

        assert len(self.vp.shape) == 2, 'vp dimension mismatch, ' \
            f'Expected 2 dimensions but got {len(self.vp.shape)} in the ' \
            f'shape of {self.vp.shape}'

        assert (
            self.vs == None 
            or
            (
                self.vp.shape == self.vs.shape 
                and self.vs.shape == self.rho.shape
            )
        ), \
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
        ).to(self.devices[0])
    
        #get offset info for optimization landscape plots
        self.ofs = kw.get('ofs', 1)

    def update(self, **kw):
        for k,v in kw.items():
            if( k in self.__slots__ ):
                setattr(self, k, v)
            else:
                warn(f'Attribute {k} does not exist...skipping ')

class DataGenerator(Data, metaclass=CombinedMeta):
    """
    DataGenerator
    ***
    This class is an abstract base class for generating seismic data. 
    It inherits from the Data class and extends it by adding a custom parameter 
        for user flexibility. 
    ***

    Attributes
    ----------
    custom : dict
        Custom parameters for user flexibility.

    Methods
    -------
    force(y, x, comp, **kw)
        Abstract method for force calculation.

    forward(**kw)
        Abstract method for forward modeling.
    """
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

    def get(self, key):
        return self.custom[key]
            
    @abstractmethod
    def force(self,p,comp,**kw):
        pass

    @abstractmethod
    def forward(self, **kw):
        pass
