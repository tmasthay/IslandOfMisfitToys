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
from typing import Callable
from abc import ABC, abstractmethod
from .custom_losses import *
from ..misfit_toys_helpers import *

plt.rc('text', usetex=False)

class Survey(metaclass=SlotMeta):
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

    src_amplitudes: Ant[
        torch.Tensor, 
        'Source amplitudes', 
        '', 
        '',
        'default: None'
    ]

    nx: Ant[int, 'Number of horizontal dofs', '1']
    ny: Ant[int, 'Number of vertical dofs', '1']
    nt: Ant[int, 'Number of time steps', '1']
    dx: Ant[float, 'Grid size horizontal', '0.0<']
    dy: Ant[float, 'Grid size vertical', '0.0<']
    dt: Ant[float, 'Time step', '0.0<']

    freq: Ant[float, 'Characteristic Ricker frequency', '0.0<']
    wavelet_amp: Ant[float, 'Characteristic source amplitude', '0.0<']
    wavelet: Ant[torch.Tensor, 'Characteristic source time signature']

    ofs: Ant[int, 'Padding for src_loc landscape', '1', 'min(nx-1,ny-1)']

    def __init__(self, **kw):
        self.devices = get_all_devices()

        #TODO: refine for GPU management
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

        #TODO: the wavelet should be entirely abstracted out!
        #   it is only used for source amplitudes which are built at beg.
        self.wavelet_amp = kw.get('wavelet_amp', 1.0)
        self.wavelet = self.wavelet_amp * deepwave.wavelets.ricker(
            kw['freq'], 
            kw['nt'], 
            kw['dt'],
            kw['peak_time']
        ).to(self.devices[0])
        self.src_amplitudes = kw.get('src_amplitudes', 
            self.wavelet \
            .repeat(self.n_shots, self.src_per_shot, 1) \
            .to(self.devices[0])
        )

        #get offset info for optimization landscape plots
        self.ofs = kw.get('ofs', 1)

    def update(self, **kw):
        for k,v in kw.items():
            if( k in self.__slots__ ):
                setattr(self, k, v)
            else:
                warn(f'Attribute {k} does not exist...skipping ')

class ModelDistributed(Survey, metaclass=CombinedMeta):
    """
    FWI class for acoustic and elastic FWI.
    """
    model: Ant[list, 'model', '', '', 'acoustic or elastic']
    u: Ant[torch.Tensor, 'Wavefield solution']
    custom: Ant[dict, 'Custom parameters for user flexibility']
    def __init__(
        self,
        *,
        model,
        **kw
    ):
        super().__init__(**kw)
        self.model = model
        self.u = None
        self.custom = dict()
        full_slots = super().__slots__ + self.__slots__
        new_keys = set(kw.keys()).difference(set(full_slots))
        for k in new_keys:
            self.custom[k] = kw[k]

    def get(self, key):
        return self.custom[key]

    def forward(self, **kw):
        """forward solver"""
        if( self.model == 'acoustic' ):
            self.u = deepwave.scalar(
                self.vp,
                self.dx,
                self.dt,
                source_amplitudes=self.src_amplitudes,
                source_locations=self.src_loc,
                receiver_locations=self.rec_loc,
                pml_freq=self.freq,
                **kw
            )[-1]
        elif( self.model.lower() == 'elastic' ):
            self.u = elastic(
                *deepwave.common.vpvsrho_to_lambmubuoyancy(
                    self.vp, 
                    self.vs, 
                    self.rho
                ),
                self.dx,
                self.dt,
                source_amplitudes_y=self.src_amplitudes,
                source_locations_y=self.src_loc,
                receiver_locations_y=self.rec_loc,
                pml_freq=self.freq,
            )[-2]
        else:
            raise ValueError(f'Unknown model type {self.model}')
        return self.u
    
    @abstractmethod
    def update_src(self, *, p, **kw):
        """update source term"""

class ModelDirac(ModelDistributed, metaclass=SlotMeta):
    def __init__(self, **kw):
        assert( 'src_amplitudes' in kw.keys() )
        super().__init__(**kw)

    def update_src(self, *, p, **kw):
        pass

class FWI(metaclass=SlotMeta):
    model: Ant[ModelDistributed, 'Abstract model', 'Must be concretized']
    loss: Ant[torch.nn.Module, 'Loss function'] 
    optimizer: Ant[torch.optim.Optimizer, 'Optimizer', 'Must be concretized']
    scheduler: Ant[torch.optim.lr_scheduler, 'Learning rate scheduler']
    n_epochs: Ant[int, 'Number of epochs', '1']
    n_batches: Ant[int, 'Number of batches', '1']

    def __init__(
        self,
        *,
        model,
        loss,
        optimizer,
        scheduler,
        n_epochs,
        n_batches
    ):
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.n_epochs = n_epochs
        self.n_batches = n_batches

    def preforward(self):
        self.optimizer.zero_grad()

    def postforward(self):
        self.optimizer.step()
        self.scheduler.step()

    def take_step(self, *, epoch, **kw):
        self.preforward()
        self.model.forward(**kw)
        self.postforward()

    def fwi(self, **kw):
        for epoch in range(self.n_epochs):
            self.take_step(epoch=epoch, **kw)
    


    