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
        src_idx_y = helper(fst_src[0], d_src[0], num_src[0])
        src_idx_x = helper(fst_src[1], d_src[1], num_src[1])
        rec_idx_y = helper(fst_rec[0], d_rec[0], num_rec[0])
        rec_idx_x = helper(fst_rec[1], d_rec[1], num_rec[1])
        self.build_src(n_shots=n_shots, idx_y=src_idx_y, idx_x=src_idx_x)
        self.build_rec(n_shots=n_shots, idx_y=rec_idx_y, idx_x=rec_idx_x)

    def build_src(
        self,
        *,
        n_shots: Ant[int, 'Number of shots'],
        idx_y: Ant[list, 'Vertical indices'],
        idx_x: Ant[list, 'Horizontal indices']
    ):
        self.src_loc = uni_src_rec(
            n_shots=n_shots, 
            idx_y=idx_y, 
            idx_x=idx_x
        )

    def build_rec(
        self,
        *,
        n_shots: Ant[int, 'Number of shots'],
        idx_y: Ant[list, 'Vertical indices'],
        idx_x: Ant[list, 'Horizontal indices']
    ):
        self.rec_loc = uni_src_rec(
            n_shots=n_shots, 
            idx_y=idx_y, 
            idx_x=idx_x
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
        
class ModelDistributed(metaclass=CombinedMeta):
    survey: Ant[Survey, 'Survey object']
    model: Ant[str, 'model', '', '', 'acoustic or elastic']
    u: Ant[torch.Tensor, 'Wavefield solution']
    custom: Ant[dict, 'Custom parameters for user flexibility']

    def __init__(
        self,
        *,
        survey,
        model,
        **kw
    ):
        self.survey = survey
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
    scheduler: Ant[torch.optim.lr_scheduler.ChainedScheduler, 'Learning rate']
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
    


    