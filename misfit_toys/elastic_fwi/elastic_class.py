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
            return dw.elastic(                vp=self.model.vp(),
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
        self.build_custom_meta(**kw)
    
    def build_custom_meta(self, **kw):
        verbosity = self.custom.get('verbosity', 1)
        print_protocol = self.custom.get('print_protocol', print)
        verbosity_levels = self.custom.get('verbosity_levels', None)
        idt_str = self.custom.get('idt_str', '    ')

        def report(s, *, idt=0, end='\n'):
            print_protocol(idt_str*idt + s, end=end)

        self.custom['run'] = run_verbosity(
            verbosity=verbosity,
            levels=verbosity_levels
        )
        self.custom['rpt'] = self.custom['run'](report)
        # def stephen_uncurry(f, _verbosity_):
        #     def helper(*args, **kw):
        #         return f(*args, _verbosity_=_verbosity_, **kw)
        #     return helper
        # rpt = self.custom['rpt']
        # self.custom['rpt_silent'] = stephen_uncurry(rpt, 'silent')
        # self.custom['rpt_progress'] = stephen_uncurry(rpt, 'progress')
        # self.custom['rpt_debug'] = stephen_uncurry(rpt, 'debug')
        # self.custom['rpt_inf'] = stephen_uncurry(rpt, 'inf')

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

    def debug_data(self, header=80*'*', extra_obj=None):
        def extract_data(obj):
            data_attr = [
                e for e in dir(obj)
                    if not e.startswith('_')           
            ]
            return {k: getattr(obj, k) for k in data_attr}
        
        def summarize(obj_name, obj):
            s = [header, f'{obj_name} data']
            if( isinstance(obj, torch.Tensor) ):
                s.append(f'    shape == {obj.shape}')
                return s
            
            data = extract_data(obj)
            runner = []
            idt = '    '
            for k,v in data.items():
                tmp = idt + k 
                if( isinstance(v, torch.Tensor) ):
                    tmp += f'.shape == {v.shape}'
                elif( isinstance(v, AbstractParam) ):
                    tmp1 = tmp + f'.param.shape == {v.param.shape}'
                    tmp2 = tmp + f'().shape == {v.forward().shape}'
                    tmp = tmp1 + '\n' + tmp2
                else:
                    tmp = idt + tmp + f'.shape == (no shape?)'
                if( tmp.startswith(2*idt) ):
                    runner.append(tmp)
                elif( '\n' in tmp ):
                    tmp = tmp.split('\n')
                    for t in tmp:
                        runner.insert(0, t)
                else:
                    runner.insert(0, tmp)
            s.extend(runner)
            return s
        
        if( extra_obj is None ): extra_obj = []
        d = [header, 'DEBUG DATA']
        d.extend(summarize('survey', self.prop.model.survey))
        d.extend(summarize('model', self.prop.model))
        for obj_name, obj in extra_obj:
            d.extend(summarize(obj_name, obj))
        d.extend(['END DEBUG DATA', header])
        return '\n'.join(d)

    def fwi(self, **kw):
        def rpt_inf(head, d):
            s = '{\n'
            for k,v in d.items():
                s += f'    {k}: {v}\n'
            s += '}\n'
            msg = head + ' = """' + s + '\n"""'
            self.custom['rpt'](msg, _verbosity_='inf')

        pre_train_meta = self.pre_train(**kw)
        rpt_inf('Pre-train kwargs', pre_train_meta)
        pre_step_meta, post_step_meta = {}, {}
        for epoch in range(self.epochs):
            pre_step_kw = {**pre_train_meta, **pre_step_meta}
            rpt_inf(f'Pre-step meta (epoch={epoch})', pre_step_meta)
            pre_step_meta = self.pre_step(epoch, **pre_step_kw)
            rpt_inf(f'Pre-step kwargs (epoch={epoch})', pre_step_kw)
            step_meta = self.take_step(epoch=epoch, **pre_step_meta)
            rpt_inf(f'Step meta (epoch={epoch})', step_meta)
            post_step_kw = {**pre_train_meta, **pre_step_meta, **step_meta}
            rpt_inf(f'Post-step kwargs (epoch={epoch})', post_step_kw)
            post_step_meta = self.post_step(epoch, **post_step_kw)
            rpt_inf(f'Post-step meta (epoch={epoch})', post_step_meta)
        post_train_kw = {**pre_train_meta, **pre_step_meta, **post_step_meta}
        rpt_inf('Post-train kwargs', post_train_kw)
        post_train_meta = self.post_train(**post_train_kw)
        rpt_inf('Post-train meta', post_train_meta)
        return post_train_meta

class FWIMetaHandler(FWIAbstract, ABC, metaclass=CombinedMeta):
    def build_custom_meta(self, **kw):
        super().build_custom_meta(**kw)
        set_field = lambda k, v: self.custom.setdefault(k, v)
        set_field('make_plots', [])
        set_field('print_freq', 1)
        set_field('cmap', 'seismic')
        set_field('aspect', 'auto')
        set_field('plot_base_path', 'plots_iomt')
        set_field('gif_speed', 100)
        set_field('plot_base_path', 'plots_iomt')
        set_field('forward_kwargs', {})

        the_time = sco('date')[0].replace(' ', '_').replace(':', '-')
        the_time = '_'.join(the_time.split('_')[1:])
        curr_run_dir = self.custom['plot_base_path'] + '/' + the_time
        self.custom['curr_run_dir'] = curr_run_dir

        os.system(f'mkdir -p {curr_run_dir}')

        def plot_curr(epoch):
            for p,do_transpose in self.custom['make_plots']:
                self.custom['rpt'](
                    f'Plotting {p} after {epoch} epochs', 
                    _verbosity_='debug'
                )
                tmp = getattr(self.prop.model, p).param
                if( do_transpose ):
                    tmp = tmp.T
                tmp1 = tmp.detach().cpu().numpy()
                plt.imshow(
                    tmp1, 
                    aspect=self.custom['aspect'], 
                    cmap=self.custom['cmap']
                )
                plt.colorbar()
                plt.title(f'{p} after {epoch} epochs')
                plt.savefig(f'{curr_run_dir}/{p}_{epoch}.jpg')
                plt.clf()
        self.custom['plot_curr'] = plot_curr
    
    def pre_train(self, **kw):
        self.custom['plot_curr'](0)
        return {'train_start': time.time()}
    
    def post_train(self, **kw):
        make_plots = self.custom['make_plots']
        curr_run_dir = self.custom['curr_run_dir']
        gif_speed = self.custom['gif_speed']
        
        def rpt(s):
            self.custom['rpt'](s, _verbosity_='debug', idt=1, end='\n')

        if( len(make_plots) > 0 ):
            for p,_ in make_plots:
                print(f'Making gif for {p}')
                os.system(
                    f'convert -delay {gif_speed} $(ls -tr ' + \
                    f'{curr_run_dir}/{p}_*.jpg) {curr_run_dir}/{p}.gif'
                )
        return {'train_end': time.time()}

    def pre_step(self, epoch, **kw):
        print_freq = self.custom['print_freq']

        rpt = self.custom['rpt']
        rpt_prog = lambda s: rpt(s, idt=0, end='\n', _verbosity_='progress')
        if( epoch % print_freq == 0 ):
            rpt_prog(f'Epoch {epoch+1}/{self.epochs}')

        return {'step_start': time.time()}

    def post_step(self, epoch, **kw): 
        self.custom['plot_curr'](epoch+1)

        rpt = self.custom['rpt']
        rpt_prog = lambda s: rpt(s, idt=1, end='\n', _verbosity_='progress')

        rpt_prog(f'Loss: {self.custom["log"]["loss"][-1]:.4e}')
        for k,v in self.custom['log']['grad_norm'].items():
            rpt_prog(f'Grad norm "{k}": {v[-1].norm():.4e}')
        epoch_time = time.time() - kw['step_start']
        total_time = time.time() - kw['train_start']
        avg_time_per_epoch = total_time / (epoch+1)
        etr = avg_time_per_epoch * (self.epochs-epoch-1)
        rpt_prog(
            f'(Epoch,Total,ETR) = ' 
                + f'({ht(epoch_time)}, {ht(total_time)}, {ht(etr)})'
        )
        return {'step_end': time.time()}   

class FWI(FWIMetaHandler, metaclass=SlotMeta):
    def take_step(self, *, epoch, **kw):
        rpt = self.custom['rpt']
        rpt_prog = lambda s: rpt(s, idt=1, end='\n', _verbosity_='progress')
        rpt_btch = lambda s,e: rpt(s, idt=1, end=e, _verbosity_='progress')
        rpt_debug = lambda s: rpt(s, idt=1, end='\n', _verbosity_='debug')

        epoch_start = kw['step_start']

        if( epoch == 0 ):
            self.custom['log'] = {
                'loss': [],
                'grad_norm': {k: [] for k in self.trainable_str}
            }
        epoch_loss = 0.0
        self.optimizer.zero_grad()
        for (batch_no, batch_idx) in enumerate(self.batches):
            batch_start = time.time()
            if( batch_no == 0 ):
                rpt_btch(f'Starting first batch of {len(self.batches)}', '\r')
            if(  batch_no > 0):
                rpt_debug(
                    self.debug_data(
                        extra_obj=[
                            ('out', out),
                            ('obs_data', self.obs_data[batch_idx])
                        ],
                    ),
                )
                rpt_debug(full_mem_report(title=f'Batch {batch_no}'))
            rpt_debug('Attempting forward solve')
            out = self.prop.forward(
                batch_idx=batch_idx, 
                **self.custom['forward_kwargs']
            )
            rpt_debug('Forward completed')
            assert( 
                out.shape == self.obs_data[batch_idx].shape,
                f'Output shape {out.shape} != ' + \
                    f'Observed data shape {self.obs_data[batch_idx].shape}'
            )
            loss_lcl = self.custom.get('loss_scaling', 1.0) \
                * self.loss(out, self.obs_data[batch_idx])
            rpt_debug('Loss computed')
            self.custom['log']['loss'].append(loss_lcl.detach().cpu())
            epoch_loss += loss_lcl.item()
            loss_lcl.backward()
            rpt_debug('Backprop done')
            if( 'clip_grad' in self.custom.keys() ):
                for att, clip_val in self.custom['clip_grad']:
                    torch.nn.utils.clip_grad_value_(
                        getattr(self.prop.model, att).param,
                        clip_val
                    )
            rpt_debug('Grad clipped')
            batch_time = time.time() - batch_start
            total_time = time.time() - epoch_start
            avg_batch_time = total_time / (batch_no+1)
            etr = avg_batch_time * (len(self.batches)-batch_no-1)
            rpt_btch(
                f'Completed {batch_no+1}/{len(self.batches)} -> '
                    + f'(batch, total, Epoch ETR) ='
                    + f' ({ht(batch_time)}, {ht(total_time)}, {ht(etr)})',
                '\r'
            )
        for name, p in zip(self.trainable_str, self.trainable):
            self.custom['log']['grad_norm'][name].append(p.grad.norm())
        self.optimizer.step()
        self.scheduler.step()
        return self.custom['log']