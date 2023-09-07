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
from ..base_helpers import human_time as ht

plt.rc('text', usetex=False)

class Survey(ABC, metaclass=CombinedMeta):
    src_loc_y: Ant[torch.Tensor, 'Source locations y']
    src_loc_x: Opt[Ant[torch.Tensor, 'Source locations x']]
    rec_loc_y: Ant[torch.Tensor, 'Receiver locations y']
    rec_loc_x: Opt[Ant[torch.Tensor, 'Receiver locations x']]
    src_amp_y: Ant[AbstractParam, 'Source amplitudes y']
    src_amp_x: Opt[Ant[AbstractParam, 'Source amplitudes x']]
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

    def to(self, device):
        self.src_loc_y = self.src_loc_y.to(device)
        self.rec_loc_y = self.rec_loc_y.to(device)
        self.src_amp_y = self.src_amp_y.to(device)
        if( self.src_loc_x is not None ):
            self.src_loc_x = self.src_loc_x.to(device)
        if( self.rec_loc_x is not None ):
            self.rec_loc_x = self.rec_loc_x.to(device)
        if( self.src_amp_x is not None ):
            self.src_amp_x = self.src_amp_x.to(device)
        return self
            
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
    def build_amp(
        self, 
        *, 
        func: Ant[Callable, 'Source amplitude function'],
        y_param: Ant[AbstractParam, 'Source amplitude y parameterization'],
        x_param: Ant[AbstractParam, 'Source amplitude x parameterization'],
        **kw: Ant[dict, 'Extensible keyword arguments']
    ):
        if( hasattr(self, 'src_loc_y') ):
            self.src_amp_y = y_param(
                param=func(self=self, pts=self.src_loc_y, comp=0, **kw),
                trainable=False,
                device='cpu'
            )
        else:
            raise ValueError('Must build source location y first!')
        if( hasattr(self, 'src_loc_x') ):
            self.src_amp_x = x_param(
                param=func(self=self, pts=self.src_loc_x, comp=1, **kw),
                trainable=False,
                device='cpu'
            )
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
        y_amp_param: Ant[AbstractParam, 'Source amplitude y parameter'],
        x_amp_param: Ant[AbstractParam, 'Source amplitude x parameter'],
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
        self.build_amp(func=amp_func, y_param=y_amp_param, x_param=x_amp_param)

class Model(metaclass=SlotMeta):
    survey: Ant[Survey, 'Survey object']
    model: Ant[str, 'model', '', '', 'acoustic or elastic']
    vp: Ant[AbstractParam, 'P-wave velocity']
    vs: Ant[AbstractParam, 'S-wave velocity']
    rho: Ant[AbstractParam, 'Density']
    ny: Ant[int, 'Number of grid points y']
    nx: Ant[int, 'Number of grid points x']
    nt: Ant[int, 'Number of time steps']
    dy: Ant[float, 'Spatial step size y']
    dx: Ant[float, 'Spatial step size x']
    dt: Ant[float, 'Temporal step size']
    freq: Ant[float, 'PML frequency']
    custom: Ant[dict, 'Custom parameters for user flexibility']

    def __init__(
        self,
        *,
        survey: Ant[Survey, 'Survey object'],
        model: Ant[str, 'model', '', '', 'acoustic or elastic'],
        vp: Ant[torch.Tensor, 'P-wave velocity'],
        vs: Opt[Ant[torch.Tensor, 'S-wave velocity']]=None,
        rho: Opt[Ant[torch.Tensor, 'Density']]=None,
        vp_param: Ant[Param, 'P-wave velocity parameter']=Param,
        vs_param: Ant[Param, 'S-wave velocity parameter']=Param,
        rho_param: Ant[Param, 'Density parameter']=Param,
        freq: Ant[float, 'PML frequency'],
        dy: Ant[float, 'Spatial step size y'],
        dx: Ant[float, 'Spatial step size x'],
        **kw
    ):
        self.survey = survey
        self.model = model
        self.vp = vp_param(param=vp, trainable=False, device='cpu')
        self.vs = vs_param(param=vs, trainable=False, device='cpu')
        self.rho = rho_param(param=rho, trainable=False, device='cpu')
        self.freq = freq
        self.dy, self.dx, self.dt = dy, dx, survey.dt
        self.ny, self.nx, self.nt = *vp.shape, survey.src_amp_y.param.shape[-1]
        self.custom = dict()
        full_slots = self.survey.__slots__ + self.__slots__
        new_keys = set(kw.keys()).difference(set(full_slots))
        for k in new_keys:
            self.custom[k] = kw[k]

    def to(self, device):
        self.vp = self.vp.to(device)
        self.vs = self.vs.to(device)
        self.rho = self.rho.to(device)
        self.survey = self.survey.to(device)
        return self

class Prop(torch.nn.Module, metaclass=SlotMeta):
    model: Opt[Ant[Model, 'Model']]

    def __init__(self, *, model, train=None, device='cpu'):
        super().__init__()
        train = {} if train is None else train
        default_train = {
            'vp': True,
            'vs': False,
            'rho': False,
            'src_amp_y': False,
            'src_amp_x': False
        }
        full_train = {**default_train, **train}
        self.model = model.to(device)
        valid_params = ['vp', 'vs', 'rho', 'src_amp_y', 'src_amp_x']
        assert all([e in valid_params for e in full_train.keys()]), \
            f'Invalid trainable parameters: expected {valid_params}' \
                + f', got {full_train.keys()}'
        self.model.vp.param.requires_grad = full_train['vp']
        self.model.vs.param.requires_grad = full_train['vs']
        self.model.rho.param.requires_grad = full_train['rho']
        self.model.survey.src_amp_y.param.requires_grad = \
            full_train['src_amp_y']
        self.model.survey.src_amp_x.param.requires_grad = \
            full_train['src_amp_x']
        self.to(device)

    def to(self, device):
        self.model = self.model.to(device)
        return self

    def forward(self, *, batch_idx, **kw):
        if( self.model.model == 'acoustic' ):
            return dw.scalar(
                self.model.vp(),
                self.model.dx,
                self.model.dt,
                source_amplitudes=self.model.survey.src_amp_y()[batch_idx],
                source_locations=self.model.survey.src_loc_y[batch_idx],
                receiver_locations=self.model.survey.rec_loc_y[batch_idx],
                **kw
            )[-1]
        elif( self.model.model == 'elastic' ):
            kw_builder = {}
            if( hasattr(self.model.survey, 'src_amp_x') ):
                kw_builder['source_amplitudes_x'] = \
                    self.model.survey.src_amp_x()
            if( hasattr(self.model.survey, 'src_loc_x') ):
                kw_builder['source_locations_x'] = \
                    self.model.survey.src_loc_x
            if( hasattr(self.model.survey, 'rec_loc_x') ):
                kw_builder['receiver_locations_x'] = \
                    self.model.survey.rec_loc_x
            full_kw = {**kw_builder, **kw}
            return dw.elastic(
                vp=self.model.vp(),
                vs=self.model.vs(),
                rho=self.model.rho(),
                dx=self.model.dx,
                dt=self.model.dt,
                source_amplitudes_y=self.model.survey.src_amp_y()[batch_idx],
                source_locations_y=self.model.survey.src_loc_y[batch_idx],
                receiver_locations_y=self.model.survey.rec_loc_y[batch_idx],
                **full_kw
            )[-2]
        else:
            raise ValueError(f'Unknown model type {self.model.model}')

class FWIAbstract(ABC, metaclass=CombinedMeta):
    prop: Ant[Prop, 'Propagator object']
    obs_data: Ant[torch.Tensor, 'Observed data']
    loss: Ant[torch.nn.Module, 'Loss function'] 
    optimizer: Ant[torch.optim.Optimizer, 'Optimizer']
    scheduler: Ant[list, 'Learning rate scheduling params']
    epochs: Ant[int, 'Number of epochs', '1']
    batches: Ant[list, 'Batches of data']
    trainable: Ant[list, 'Trainable parameters']
    trainable_str: Ant[list, 'Trainable parameters as strings']
    custom: Ant[dict, 'Custom parameters for user flexibility']

    def __init__(
        self,
        *,
        prop,
        obs_data,
        loss,
        optimizer,
        scheduler,
        epochs,
        num_batches,
        multi_gpu=False,
        **kw
    ):
        if( multi_gpu ):
            self.prop = torch.nn.DataParallel(prop).to('cuda')
        else:
            self.prop = prop
        self.obs_data = obs_data
        self.loss = loss
        self.epochs = epochs
        self.batches = self.build_batches(num_batches, obs_data.shape[0])
        self.build_trainables()
        self.optimizer = optimizer[0](self.trainable, **optimizer[1])
        self.build_scheduler(scheduler)
        self.build_customs(**kw)

    def build_batches(self, num_batches, n_data):
        return np.array_split(np.arange(n_data), num_batches)

    def build_trainables(self):
        self.trainable_str = []
        self.trainable = []
        for s in ['vp', 'vs', 'rho']:
            curr = getattr(self.prop.model, s)
            if( curr.param.requires_grad ):
                self.trainable_str.append(s)
                self.trainable.append(curr.param)

        for s in ['src_amp_y', 'src_amp_x']:
            curr = getattr(self.prop.model.survey, s)
            if( curr.param.requires_grad ):
                self.trainable_str.append(s)
                self.trainable.append(curr.param)
    
    def build_scheduler(self, scheduler):
        schedule_list = []
        for s in scheduler:
            schedule_list.append(s[0](self.optimizer, **s[1]))
        self.scheduler = torch.optim.lr_scheduler.ChainedScheduler(
            schedule_list
        )

    def build_customs(self, **kw):
        self.custom = {}
        for k,v in kw.items():
            self.custom[k] = v

    @abstractmethod
    def take_step(self, **kw): pass

    @abstractmethod
    def pre_train(self, **kw): pass

    @abstractmethod
    def post_train(self, **kw): pass

    @abstractmethod
    def pre_step(self, epoch, **kw): pass

    @abstractmethod
    def post_step(self, epoch, **kw): pass

    def fwi(self, **kw):
        pre_train_meta = self.pre_train(**kw)
        pre_step_meta, post_step_meta = {}, {}
        for epoch in range(self.epochs):
            pre_step_kw = {**pre_train_meta, **pre_step_meta}
            pre_step_meta = self.pre_step(epoch, **pre_step_kw)
            step_meta = self.take_step(epoch=epoch, **kw)
            post_step_kw = {**pre_train_meta, **pre_step_meta, **step_meta}
            post_step_meta = self.post_step(epoch, **post_step_kw)
        post_train_kw = {**pre_train_meta, **pre_step_meta, **post_step_meta}
        post_train_meta = self.post_train(**post_train_kw)
        return post_train_meta

#think of better name later on
# class FWIConcrete1(FWIAbstract, metaclass=CombinedMeta):

#     @abstractmethod
#     def take_step(self, **kw):
#         pass

#     def pre_process(self):
#         make_plots = self.custom.get('make_plots', [])
#         verbose = self.custom.get('verbose', False)
#         print_freq = self.custom.get('print_freq', 1)
#         cmap = self.custom.get('cmap', 'seismic')
#         aspect = self.custom.get('aspect', 'auto')
#         plot_base_path = self.custom.get('plot_base_path', 'plots_iomt')
#         gif_speed = self.custom.get('gif_speed', 100)

#         the_time = sco('date')[0].replace(' ', '_').replace(':', '-')
#         the_time = '_'.join(the_time.split('_')[1:])
#         curr_run_dir = f'{plot_base_path}/{the_time}'
#         os.system(f'mkdir -p {curr_run_dir}')
#         def plot_curr(epoch):
#             for p,do_transpose in make_plots:
#                 tmp = getattr(self.model, p)
#                 if( do_transpose ):
#                     tmp = tmp.T
#                 tmp1 = tmp.detach().cpu().numpy()
#                 plt.imshow(tmp1, aspect=aspect, cmap=cmap)
#                 plt.colorbar()
#                 plt.title(f'{p} after {epoch} epochs')
#                 plt.savefig(f'{curr_run_dir}/{p}_{epoch}.jpg')
#                 plt.clf()
#         plot_curr(0)
#         return {
#             'verbose': verbose,
#             'print_freq': print_freq,
#             'curr_run_dir': curr_run_dir,
#             'gif_speed': gif_speed,
#             'plot_curr': plot_curr,
#             'make_plots': make_plots
#         }
    
#     def in_loop_pre_process(self, **kw):
#         def helper(epoch):
#             if( kw['verbose'] and epoch % kw['print_freq'] == 0 ):
#                 print(f'Epoch {epoch+1}/{self.epochs}')
#         return helper

#     def in_loop_post_process(self, **kw):
#         plot_curr = kw['plot_curr']
#         def helper(epoch):
#             plot_curr(epoch+1)
#         return helper

#     def post_process(self, **kw):
#         make_plots = kw['make_plots']
#         curr_run_dir = kw['curr_run_dir']
#         gif_speed = kw['gif_speed']
#         if( len(make_plots) > 0 ):
#             for p,_ in make_plots:
#                 print(f'Making gif for {p}')
#                 os.system(
#                     f'convert -delay {gif_speed} $(ls -tr ' + \
#                     f'{curr_run_dir}/{p}_*.jpg) {curr_run_dir}/{p}.gif'
#                 )
        
#     def fwi(self, **kw):
#         kw_default = {
#             'source_amplitudes': self.model.survey.src_amp_y,
#             'source_locations': self.model.survey.src_loc_y,
#             'receiver_locations': self.model.survey.rec_loc_y
#         }
#         kw = {**kw_default, **kw}
#         precomputed_meta = self.pre_process()
#         in_loop_pre_process = self.in_loop_pre_process(**precomputed_meta)
#         in_loop_post_process = self.in_loop_post_process(**precomputed_meta)
#         start_time = time.time()
#         for epoch in range(self.epochs):
#             start_epoch = time.time()
#             in_loop_pre_process(epoch=epoch)
#             loss_lcl, grad_norms = self.take_step(epoch=epoch, **kw)
#             print_idt = lambda x,idt: print('%s%s'%(4*idt*' ',x))
#             print_idt(f'Loss: {loss_lcl:.4e}', 1)
#             print_idt(
#                 f'Learning rate: {self.optimizer.param_groups[0]["lr"]:.4e}', 
#                 1
#             )
#             for (i,p) in enumerate(self.trainable):
#                 name = get_member_name(self.model, p)
#                 print_idt(f'Grad norm "{name}": {grad_norms[i]:.4e}', 1)
#             in_loop_post_process(epoch=epoch)
#             epoch_time = time.time() - start_epoch
#             total_time = time.time() - start_time
#             avg_time_per_epoch = total_time / (epoch+1)
#             etr = avg_time_per_epoch * (self.epochs-epoch-1)
#             print_idt(f'Epoch time: {epoch_time:.4e} s', 2)
#             print_idt(f'Avg time per epoch: {avg_time_per_epoch:.4e} s', 2)
#             print_idt(f'ETR: {etr:.4e} s', 2)
#         self.post_process(**precomputed_meta)
    
class FWIMetaHandler(FWIAbstract, ABC, metaclass=CombinedMeta):
    def pre_train(self, **kw):
        make_plots = self.custom.get('make_plots', [])
        verbose = self.custom.get('verbose', False)
        print_freq = self.custom.get('print_freq', 1)
        cmap = self.custom.get('cmap', 'seismic')
        aspect = self.custom.get('aspect', 'auto')
        plot_base_path = self.custom.get('plot_base_path', 'plots_iomt')
        gif_speed = self.custom.get('gif_speed', 100)

        rpt = report(verbose)

        the_time = sco('date')[0].replace(' ', '_').replace(':', '-')
        the_time = '_'.join(the_time.split('_')[1:])
        curr_run_dir = f'{plot_base_path}/{the_time}'
        os.system(f'mkdir -p {curr_run_dir}')
        def plot_curr(epoch):
            for p,do_transpose in make_plots:
                rpt(f'Plotting {p} after {epoch} epochs', 1)
                tmp = getattr(self.prop.model, p).param
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
            'print_freq': print_freq,
            'curr_run_dir': curr_run_dir,
            'gif_speed': gif_speed,
            'plot_curr': plot_curr,
            'make_plots': make_plots,
            'global_start': time.time(),
            **self.custom
        }

    def post_train(self, **kw):
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
        return {}

    def pre_step(self, epoch, **kw):
        verbose = kw.get('verbose', True)
        print_freq = kw.get('print_freq', 1)
        if( verbose and epoch % print_freq == 0 ):
            print(f'Epoch {epoch+1}/{self.epochs}')
        return {'step_start': time.time()}

    def post_step(self, epoch, **kw): 
        kw['plot_curr'](epoch+1)
        verbose = self.custom.get('verbose', False)
        rpt = report(verbose)
        idt = 1
        rpt(f'Loss: {self.custom["log"]["loss"][-1]:.4e}', idt)
        for k,v in self.custom['log']['grad'].items():
            rpt(f'Grad norm "{k}": {v[-1].norm():.4e}', idt)
        epoch_time = time.time() - kw['step_start']
        total_time = time.time() - kw['global_start']
        avg_time_per_epoch = total_time / (epoch+1)
        etr = avg_time_per_epoch * (self.epochs-epoch-1)
        rpt(
            f'(Epoch,Total,ETR) = ' 
                + f'({ht(epoch_time)}, {ht(total_time)}, {ht(etr)})', 
            idt
        )
        return {}
        

class FWI(FWIMetaHandler, metaclass=SlotMeta):
    def take_step(self, *, epoch, **kw):
        if( epoch == 0 ):
            self.custom['log'] = {
                'loss': [],
                'grad_norm': {k: [] for k in self.trainable_str}
            }
        epoch_loss = 0.0
        self.optimizer.zero_grad()
        for batch_idx in self.batches:
            out = self.prop.forward(batch_idx=batch_idx, **kw)
            loss_lcl = self.custom.get('loss_scaling', 1.0) \
                * self.loss(out, self.obs_data)
            self.custom['log']['loss'].append(loss_lcl.detach().cpu())
            epoch_loss += loss_lcl.item()
            loss_lcl.backward()
            if( 'clip_grad' in self.custom.keys() ):
                for att, clip_val in self.custom['clip_grad']:
                    torch.nn.utils.clip_grad_value_(
                        getattr(self.prop.model, att).param,
                        clip_val
                    )
        for name, p in zip(self.trainable_str, self.trainable):
            self.custom['log']['grad_norm'][name].append(p.grad.norm())
        self.optimizer.step()
        self.scheduler.step()
        return self.custom['log']