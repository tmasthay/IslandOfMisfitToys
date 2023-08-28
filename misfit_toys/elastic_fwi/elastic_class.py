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
from typing import Annotated as Ant
from typing import Optional as Opt
from .custom_losses import *
from ..misfit_toys_helpers import *

plt.rc('text', usetex=False)

class Survey(ABC, metaclass=CombinedMeta):
    src_loc: Ant[torch.Tensor, 'Source locations']
    rec_loc: Ant[torch.Tensor, 'Receiver locations']
    src_amp_y: Ant[torch.Tensor, 'Source amplitudes y']
    src_amp_x: Opt[Ant[torch.Tensor, 'Source amplitudes x']]
    custom: Ant[dict, 'Custom parameters for user flexibility']

    def __init__(self, **kw):
        raise NotImplementedError('Survey constructor not concretized!')

    def update(self, **kw):
        for k,v in kw.items():
            if( k in self.__slots__ ):
                setattr(self, k, v)
            else:
                raise ValueError(f'Attribute {k}, slots={self.__slots__}')
    
    def update_custom(self, **kw):
        for k,v in kw.items():
            if( k in self.__slots__ ):
                raise ValueError(
                    f'Attribute {k}, slots={self.__slots__}' + \
                    '...use update instead'
                )
            self.custom[k] = v

    @abstractmethod
    def build_src(self, **kw):
        """build source locations"""

    @abstractmethod
    def build_rec(self, **kw):
        """build receiver locations"""

    @abstractmethod
    def build_amp(self, **kw):
        """build source amplitudes"""

class SurveyUniformAbstract(Survey, metaclass=CombinedMeta):
    def _init_uniform_(
        self,
        *,
        n_shots: Ant[int, 'Number of shots'],
        fst_src: Ant[list, 'First source location'],
        d_src: Ant[list, 'Source spacing'],
        num_src: Ant[list, 'Number of sources'],
        fst_rec: Ant[list, 'First receiver location'],
        d_rec: Ant[list, 'Receiver spacing'],
        num_rec: Ant[list, 'Number of receivers'],
    ):
        helper = lambda fst, d, num: [fst + i * d for i in range(num)]
        src_idx = []
        rec_idx = []
        assert( len(fst_src) in [1,2] )
        for i in range(len(fst_src)):
            assert(len(fst_src[i]) == 2)
            tmp_src = []
            tmp_rec = []
            for j in range(len(fst_src[i])):
                tmp_src.append(
                    helper(fst_src[i][j], d_src[i][j], num_src[i][j])
                )
                tmp_rec.append(    
                    helper(fst_rec[i][j], d_rec[i][j], num_rec[i][j])
                )
            src_idx.append(tmp_src)
            rec_idx.append(tmp_rec)
        self.build_src(n_shots=n_shots, idx=src_idx)
        self.build_rec(n_shots=n_shots, idx=rec_idx)

    def build_src(
        self,
        *,
        n_shots: Ant[int, 'Number of shots'],
        idx: Ant[list, 'Source indices']
    ):
        assert( len(idx) in [1,2] )
        assert( len(idx[0]) == 2 )
        self.src_loc_y = uni_src_rec(
            n_shots=n_shots,
            idx_y=idx[0][0],
            idx_x=idx[0][1]
        )
        if( len(idx) == 2 ):
            assert(len(idx[1]) == 2)
            self.src_loc_x = uni_src_rec(
                n_shots=n_shots,
                idx_y=idx[1][0],
                idx_x=idx[1][1]
            )

    def build_rec(
        self,
        *,
        n_shots: Ant[int, 'Number of shots'],
        idx: Ant[list, 'Source indices']
    ):
        assert( len(idx) in [1,2] )
        assert( len(idx[0]) == 2 )
        self.rec_loc_y = uni_src_rec(
            n_shots=n_shots,
            idx_y=idx[0][0],
            idx_x=idx[0][1]
        )
        if( len(idx) == 2 ):
            assert(len(idx[1]) == 2)
            self.rec_loc_x = uni_src_rec(
                n_shots=n_shots,
                idx_y=idx[1][0],
                idx_x=idx[1][1]
            )

class SurveyFunctionAmpAbstract(Survey, metaclass=CombinedMeta):
    def build_amp(self, *, func, **kw):
        if( 'comp' not in kw.keys() ):
            self.src_amp_y = func(self.src_loc, comp=0)
        else:
            if( 0 in kw['comp'] ):
                self.src_amp_y = func(self.src_loc, comp=0)
            elif( 1 in kw['comp'] ):
                self.src_amp_x = func(self.src_loc, comp=1)
            else:
                raise ValueError('Invalid kwargs to build_amp: %s'%str(kw))
        if( not hasattr(self, 'src_amp_x') ):
            self.src_amp_x = None

class SurveyUniformLambda(
    SurveyUniformAbstract,
    SurveyFunctionAmpAbstract,
    metaclass=SlotMeta
):
    def __init__(
        self,
        *,
        n_shots: Ant[int, 'Number of shots'],
        fst_src: Ant[list, 'First source location'],
        d_src: Ant[list, 'Source spacing'],
        num_src: Ant[list, 'Number of sources'],
        fst_rec: Ant[list, 'First receiver location'],
        d_rec: Ant[list, 'Receiver spacing'],
        num_rec: Ant[list, 'Number of receivers'],
        amp_func
    ):
        super()._init_uniform_(
            n_shots=n_shots,
            fst_src=fst_src,
            d_src=d_src,
            num_src=num_src,
            fst_rec=fst_rec,
            d_rec=d_rec,
            num_rec=num_rec
        )
        self.build_amp(func=amp_func)
        non_essential = [('src_amp_x', None), ('custom', dict())]
        for k,v in non_essential:
            if( not hasattr(self, k) ):
                setattr(self, k, v)
        
class Model(metaclass=SlotMeta):
    survey: Ant[Survey, 'Survey object']
    model: Ant[str, 'model', '', '', 'acoustic or elastic']
    u: Ant[torch.Tensor, 'Wavefield solution']
    vp: Ant[torch.Tensor, 'P-wave velocity']
    vs: Ant[torch.Tensor, 'S-wave velocity']
    rho: Ant[torch.Tensor, 'Density']
    dx: Ant[float, 'Spatial step size']
    dt: Ant[float, 'Temporal step size']
    freq: Ant[float, 'PML frequency']
    custom: Ant[dict, 'Custom parameters for user flexibility']

    def __init__(
        self,
        *,
        survey,
        model,
        vp,
        vs=None,
        rho=None,
        freq,
        dt,
        **kw
    ):
        self.survey = survey
        self.model = model
        self.u = None
        self.vp = vp
        self.vs = vs
        self.rho = rho
        self.freq = freq
        self.dy, self.dx, self.dt = *vp.shape, dt
        self.nt = survey.src_amp_y.shape[-1]
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
                source_amplitudes=self.src_amp_y,
                source_locations=self.src_loc,
                receiver_locations=self.rec_loc,
                pml_freq=self.freq,
                **kw
            )[-1]
        elif( self.model.lower() == 'elastic' ):
            kw_lcl = {
                'source_amplitudes_y': self.src_amp_y,
                'source_locations_y': self.src_loc,
                'receiver_locations_y': self.rec_loc,
                'pml_freq': self.freq
            }
            if( hasattr(self.survey, 'src_amp_x') ):
                kw_lcl['source_amplitudes_x'] = self.src_amp_x

            self.u = elastic(
                *deepwave.common.vpvsrho_to_lambmubuoyancy(
                    self.vp, 
                    self.vs, 
                    self.rho
                ),
                self.dx,
                self.dt,
                **kw_lcl
            )[-2]
        else:
            raise ValueError(f'Unknown model type {self.model}')
        return self.u

class FWIAbstract(ABC, metaclass=CombinedMeta):
    model: Ant[Model, 'Model']
    obs_data: Ant[torch.Tensor, 'Observed data']
    loss: Ant[torch.nn.Module.loss, 'Loss function'] 
    optimizer: Ant[torch.optim.Optimizer, 'Optimizer']
    scheduler: Ant[torch.optim.lr_scheduler.ChainedScheduler, 'Learning rate']
    n_epochs: Ant[int, 'Number of epochs', '1']
    n_batches: Ant[int, 'Number of batches', '1']
    trainable: Ant[list, 'Trainable parameters']

    def __init__(
        self,
        *,
        model,
        loss,
        optimizer,
        scheduler,
        n_epochs,
        n_batches,
        trainable
    ):
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.n_epochs = n_epochs
        self.n_batches = n_batches
        self.trainable = trainable
        for p in self.trainable:
            if( hasattr(self.model, p) ):
                att = getattr(self.model, p)
                setattr(att.requires_grad, True)
            else:
                raise ValueError(f'Unknown trainable parameter {p}')

    @abstractmethod
    def take_step(self, **kw):
        pass

    def fwi(self, **kw):
        for epoch in range(self.n_epochs):
            self.take_step(**kw)
    
class FWI(FWIAbstract, metaclass=SlotMeta):
    def take_step(self, **kw):
        self.optimizer.zero_grad()
        self.model.forward(**kw)
        loss_lcl = self.loss(self.model.u, self.obs_data)
        loss_lcl.backward()
        self.optimizer.step()
        self.scheduler.step()

    


    