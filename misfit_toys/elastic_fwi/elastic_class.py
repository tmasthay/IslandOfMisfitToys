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
from tqdm import tqdm
from datetime import datetime

plt.rc('text', usetex=False)

class Survey(ABC, metaclass=CombinedMeta):
    src_loc_y: Ant[torch.Tensor, 'Source locations y']
    src_loc_x: Opt[Ant[torch.Tensor, 'Source locations x']]
    rec_loc_y: Ant[torch.Tensor, 'Receiver locations y']
    rec_loc_x: Opt[Ant[torch.Tensor, 'Receiver locations x']]
    src_amp_y: Ant[torch.Tensor, 'Source amplitudes y']
    src_amp_x: Opt[Ant[torch.Tensor, 'Source amplitudes x']]
    dt: Ant[float, 'Temporal step size']
    nt: Ant[int, 'Number of time steps']
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
        src_y: Ant[dict, 'Source y info']=None,
        src_x: Ant[dict, 'Source x info']=None,
        rec_y: Ant[dict, 'Rec y info']=None,
        rec_x: Ant[dict, 'Rec x info']=None
    ):
        get_field = lambda meth, d: None if d is None \
            else meth(n_shots=n_shots, **d)

        get_src = lambda d: get_field(self.build_src, d)
        get_rec = lambda d: get_field(self.build_rec, d)

        self.src_loc_y = get_src(src_y)
        self.src_loc_x = get_src(src_x)
        self.rec_loc_y = get_rec(rec_y)
        self.rec_loc_x = get_rec(rec_x)
 
    def build_src(
        self,
        *,
        n_shots: Ant[int, 'Number of shots'],
        src_per_shot: Ant[int, 'Sources per shot'],
        fst_src: Ant[list, 'First source location'],
        src_depth: Ant[float, 'Source depth'],
        d_src: Ant[float, 'Source spacing'],
        d_intra_shot: Ant[float, 'Intra-shot spacing']
    ):
        return towed_src(
            n_shots=n_shots,
            src_per_shot=src_per_shot,
            fst_src=fst_src,
            src_depth=src_depth,
            d_src=d_src,
            d_intra_shot=d_intra_shot,
        )

    def build_rec(
        self,
        *,
        n_shots: Ant[int, 'Number of shots'],
        rec_per_shot: Ant[int, 'Receivers per shot'],
        fst_rec: Ant[list, 'First receiver location'],
        rec_depth: Ant[float, 'Receiver depth'],
        d_rec: Ant[float, 'Receiver spacing'],
    ):
        return fixed_rec(
            n_shots=n_shots,
            rec_per_shot=rec_per_shot,
            fst_rec=fst_rec,
            rec_depth=rec_depth,
            d_rec=d_rec
        )

class SurveyFunctionAmpAbstract(Survey, metaclass=CombinedMeta):
    def build_amp(self, *, func, **kw):
        if( hasattr(self, 'src_loc_y') ):
            self.src_amp_y = func(self=self, pts=self.src_loc_y, comp=0, **kw)
        else:
            raise ValueError('Must build source location y first!')
        if( hasattr(self, 'src_loc_x') ):
            self.src_amp_x = func(self=self, pts=self.src_loc_x, comp=1, **kw)
        else:
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
        src_y: Ant[dict, 'Source y info']=None,
        src_x: Ant[dict, 'Source x info']=None,
        rec_y: Ant[dict, 'Rec y info']=None,
        rec_x: Ant[dict, 'Rec x info']=None,
        amp_func: Ant[Callable, 'Source amplitude function'],
        deploy: Ant[list, 'GPU/CPU Deployment protocol'],
        nt: Ant[int, 'Number of time steps'],
        dt: Ant[float, 'Temporal step size'],
        **kw
    ):
        super()._init_uniform_(
            n_shots=n_shots,
            src_y=src_y,
            src_x=src_x,
            rec_y=rec_y,
            rec_x=rec_x
        )
        self.nt = nt
        self.dt = dt
        set_nonslotted_params(self, kw)
        self.build_amp(func=amp_func)
        device_deploy(self, deploy)

class Model(metaclass=SlotMeta):
    survey: Ant[Survey, 'Survey object']
    model: Ant[str, 'model', '', '', 'acoustic or elastic']
    u: Ant[torch.Tensor, 'Wavefield solution']
    vp: Ant[torch.Tensor, 'P-wave velocity']
    vs: Ant[torch.Tensor, 'S-wave velocity']
    rho: Ant[torch.Tensor, 'Density']
    ny: Ant[int, 'Number of grid points y']
    nx: Ant[int, 'Number of grid points x']
    nt: Ant[int, 'Number of time steps']
    dy: Ant[float, 'Spatial step size y']
    dx: Ant[float, 'Spatial step size x']
    dt: Ant[float, 'Temporal step size']
    freq: Ant[float, 'PML frequency']
    custom: Ant[dict, 'Custom parameters for user flexibility']
    propagator: Ant[Callable, 'Propagator function']

    def __init__(
        self,
        *,
        survey: Ant[Survey, 'Survey object'],
        model: Ant[str, 'model', '', '', 'acoustic or elastic'],
        vp: Ant[torch.Tensor, 'P-wave velocity'],
        vs: Opt[Ant[torch.Tensor, 'S-wave velocity']]=None,
        rho: Opt[Ant[torch.Tensor, 'Density']]=None,
        freq: Ant[float, 'PML frequency'],
        dy: Ant[float, 'Spatial step size y'],
        dx: Ant[float, 'Spatial step size x'],
        deploy: Ant[list, 'GPU/CPU Deployment protocol'],
        multi_gpu: Ant[bool, 'Multi-GPU mode']=False,
        **kw
    ):
        self.survey = survey
        self.model = model
        self.u = None
        self.vp = vp
        self.vs = vs
        self.rho = rho
        self.freq = freq
        self.dy, self.dx, self.dt = dy, dx, survey.dt
        self.ny, self.nx, self.nt = *vp.shape, survey.src_amp_y.shape[-1]
        self.custom = dict()
        full_slots = self.survey.__slots__ + self.__slots__
        new_keys = set(kw.keys()).difference(set(full_slots))
        for k in new_keys:
            self.custom[k] = kw[k]
        device_deploy(self, deploy)
        self.setup_multi_gpu(multi_gpu=multi_gpu)

    def setup_multi_gpu(self, *, multi_gpu):
        if( multi_gpu ):
            if( self.model == 'acoustic' ):
                self.propagator = torch.nn.DataParallel(
                    deepwave.Scalar(self.vp, self.dx)
                )
            else:
                raise NotImplementedError('Elastic multi-GPU not implemented')
        else:
            if( self.model.model == 'acoustic' ):
                self.propagator = lambda dtd, sad, sld, rld, pmlf: \
                    deepwave.scalar(
                        self.vp,
                        self.dx,
                        dtd,
                        source_amplitudes=sad,
                        source_locations=sld,
                        receiver_locations=rld,
                        pml_freq=pmlf
                    )[-1]
            else:
                self.propagator = lambda dtd, sad, sld, rld, pmlf: \
                    elastic(
                        *deepwave.common.vpvsrho_to_lambmubuoyancy(
                            self.vp, 
                            self.vs, 
                            self.rho
                        ),
                        self.dx,
                        dtd,
                        source_amplitudes_y=sad,
                        source_locations_y=sld,
                        receiver_locations_y=rld,
                        pml_freq=pmlf
                    )[-2]

    def get(self, key):
        return self.custom[key]

    def forward(self, **kw):
        """forward solver"""
        if( self.model == 'acoustic' ):
            self.u = self.propagator.forward(
                self.dt,
                source_amplitudes=self.survey.src_amp_y,
                source_locations=self.survey.src_loc_y,
                receiver_locations=self.survey.rec_loc_y,
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

            self.u = self.propagator.forward(
                self.dt,
                **kw_lcl
            )[-2]
        else:
            raise ValueError(f'Unknown model type {self.model}')
        return self.u

class FWIAbstract(ABC, metaclass=CombinedMeta):
    model: Ant[Model, 'Model']
    obs_data: Ant[torch.Tensor, 'Observed data']
    loss: Ant[torch.nn.Module, 'Loss function'] 
    optimizer: Ant[torch.optim.Optimizer, 'Optimizer']
    scheduler: Ant[list, 'Learning rate scheduling params']
    epochs: Ant[int, 'Number of epochs', '1']
    batch_size: Ant[int, 'Batch size', '1']
    trainable: Ant[list, 'Trainable parameters']
    custom: Ant[dict, 'Custom parameters for user flexibility']

    def __init__(
        self,
        *,
        model,
        obs_data,
        loss,
        optimizer,
        scheduler,
        epochs,
        batch_size,
        trainable,
        deploy: Ant[list, 'GPU/CPU Deployment protocol'],
        **kw
    ):
        self.model = model
        self.obs_data = obs_data
        self.loss = loss
        self.epochs = epochs
        self.batch_size = batch_size
        self.trainable = []
        for p in trainable:
            if( hasattr(self.model, p) ):
                att = getattr(self.model, p)
                setattr(att, 'requires_grad', True)
                self.trainable.append(att)
            else:
                raise ValueError(f'Unknown trainable parameter {p}')
        self.optimizer = optimizer[0](self.trainable, **optimizer[1])

        schedule_list = []
        for s in scheduler:
            schedule_list.append(s[0](self.optimizer, **s[1]))
        self.scheduler = torch.optim.lr_scheduler.ChainedScheduler(
            schedule_list
        )

        self.custom = dict()
        for k,v in kw.items():
            self.custom[k] = v
        device_deploy(self, deploy)

    @abstractmethod
    def take_step(self, **kw):
        pass

    def pre_process(self):
        make_plots = self.custom.get('make_plots', [])
        verbose = self.custom.get('verbose', False)
        print_freq = self.custom.get('print_freq', 1)
        cmap = self.custom.get('cmap', 'seismic')
        aspect = self.custom.get('aspect', 'auto')
        plot_base_path = self.custom.get('plot_base_path', 'plots_iomt')
        gif_speed = self.custom.get('gif_speed', 100)

        the_time = sco('date')[0].replace(' ', '_').replace(':', '-')
        the_time = '_'.join(the_time.split('_')[1:])
        curr_run_dir = f'{plot_base_path}/{the_time}'
        os.system(f'mkdir -p {curr_run_dir}')
        def plot_curr(epoch):
            for p,do_transpose in make_plots:
                tmp = getattr(self.model, p)
                if( do_transpose ):
                    tmp = tmp.T
                tmp1 = tmp.detach().cpu().numpy()
                plt.imshow(tmp1, aspect=aspect, cmap=cmap)
                plt.colorbar()
                plt.title(f'{p} after {epoch} epochs')
                plt.savefig(f'{curr_run_dir}/{p}_{epoch}.jpg')
                plt.clf()
        plot_curr(0)
        return {
            'verbose': verbose,
            'print_freq': print_freq,
            'curr_run_dir': curr_run_dir,
            'gif_speed': gif_speed,
            'plot_curr': plot_curr,
            'make_plots': make_plots
        }
    
    def in_loop_pre_process(self, **kw):
        def helper(epoch):
            if( kw['verbose'] and epoch % kw['print_freq'] == 0 ):
                print(f'Epoch {epoch+1}/{self.epochs}')
        return helper

    def in_loop_post_process(self, **kw):
        plot_curr = kw['plot_curr']
        def helper(epoch):
            plot_curr(epoch+1)
        return helper

    def post_process(self, **kw):
        make_plots = kw['make_plots']
        curr_run_dir = kw['curr_run_dir']
        gif_speed = kw['gif_speed']
        if( len(make_plots) > 0 ):
            for p,_ in make_plots:
                print(f'Making gif for {p}')
                os.system(
                    f'convert -delay {gif_speed} $(ls -tr ' + \
                    f'{curr_run_dir}/{p}_*.jpg) {curr_run_dir}/{p}.gif'
                )
        
    def fwi(self, **kw):
        precomputed_meta = self.pre_process()
        in_loop_pre_process = self.in_loop_pre_process(**precomputed_meta)
        in_loop_post_process = self.in_loop_post_process(**precomputed_meta)
        start_time = time.time()
        for epoch in range(self.epochs):
            start_epoch = time.time()
            in_loop_pre_process(epoch=epoch)
            loss_lcl, grad_norms = self.take_step(epoch=epoch, **kw)
            print_idt = lambda x,idt: print('%s%s'%(4*idt*' ',x))
            print_idt(f'Loss: {loss_lcl:.4e}', 1)
            print_idt(
                f'Learning rate: {self.optimizer.param_groups[0]["lr"]:.4e}', 
                1
            )
            for (i,p) in enumerate(self.trainable):
                name = get_member_name(self.model, p)
                print_idt(f'Grad norm "{name}": {grad_norms[i]:.4e}', 1)
            in_loop_post_process(epoch=epoch)
            epoch_time = time.time() - start_epoch
            total_time = time.time() - start_time
            avg_time_per_epoch = total_time / (epoch+1)
            etr = avg_time_per_epoch * (self.epochs-epoch-1)
            print_idt(f'Epoch time: {epoch_time:.4e} s', 2)
            print_idt(f'Avg time per epoch: {avg_time_per_epoch:.4e} s', 2)
            print_idt(f'ETR: {etr:.4e} s', 2)
        self.post_process(**precomputed_meta)
    
class FWI(FWIAbstract, metaclass=SlotMeta):
    def take_step(self, *, epoch, **kw):
        self.optimizer.zero_grad()
        self.model.forward(**kw)
        loss_lcl = self.custom.get('loss_scaling', 1.0) \
            * self.loss(self.model.u, self.obs_data)
        loss_lcl.backward()
        if( 'clip_grad' in self.custom.keys() ):
            for att, clip_val in self.custom['clip_grad']:
                torch.nn.utils.clip_grad_value_(
                    getattr(self.model, att),
                    clip_val
                )
        grad_norms = []
        for p in self.trainable:
            grad_norms.append(p.grad.norm())
        self.optimizer.step()
        self.scheduler.step()
        return loss_lcl, grad_norms