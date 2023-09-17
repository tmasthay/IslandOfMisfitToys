import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
import deepwave
from deepwave import elastic
import numpy as np
import argparse
from random import randint
from warnings import warn
from typing import Callable, Union, Optional as Opt, Annotated as Ant
from abc import ABC, abstractmethod
from .custom_losses import *
from ..utils import *
from tqdm import tqdm
from datetime import datetime
from ..swiffer import human_time as ht
import sys
from pympler.asizeof import asizeof

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
    model: Ant[Model, 'Model']

    def __init__(self, *, model, train=None):
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
        self.model = model
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

    def get_elastic_kwargs(self, idx, device):
        if( self.model != 'elastic' ):
            raise ValueError('No elastic kwargs for acoustic model')
        else:
            kw = {}
            if( hasattr(self.model.survey, 'src_amp_x') ):
                kw['source_amplitudes_x'] = \
                    self.model.survey.src_amp_x(idx=idx).to(device)
                kw['source_locations_x'] = \
                    self.model.survey.src_loc_x[idx].to(device)
                kw['receiver_locations_x'] = \
                    self.model.survey.rec_loc_x[idx].to(device)
            return kw

    def forward(self):
        if( self.model.model == 'acoustic' ):
            v = self.model.vp()
            amp_y = self.model.survey.src_amp_y(),
            srcy = self.model.survey.src_loc_y,
            recy = self.model.survey.rec_loc_y
            input(v.shape)
            input(amp_y.shape)
            input(srcy.shape)
            input(recy.shape)
            return dw.scalar(
                v,
                self.model.dx,
                self.model.dt,
                source_amplitudes=amp_y,
                source_locations=srcy,
                receiver_locations=recy,
            )[-1]
        elif( self.model.model == 'elastic' ):
            full_kw = {**self.get_elastic_kwargs}
            return dw.elastic(
                vp=self.model.vp(),
                vs=self.model.vs(),
                rho=self.model.rho(),
                dx=self.model.dx,
                dt=self.model.dt,
                source_amplitudes_y=self.model.survey.src_amp_y(),
                source_locations_y=self.model.survey.src_loc_y,
                receiver_locations_y=self.model.survey.rec_loc_y
                **full_kw
            )[-2]
        else:
            raise ValueError(f'Unknown model type {self.model.model}')
        
class FWIAbstract(ABC, metaclass=CombinedMeta):
    prop: Ant[Prop, 'Propagator']
    obs_data: Ant[torch.Tensor, 'Observed data']
    loss: Ant[torch.nn.Module, 'Loss function'] 
    optimizer: Ant[torch.optim.Optimizer, 'Optimizer']
    scheduler: Ant[list, 'Learning rate scheduling params']
    epochs: Ant[int, 'Number of epochs', '1']
    num_batches: Ant[list, 'Batches of data']
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
        num_batches=None,
        batch_size=None,
        **kw
    ):
        self.prop = prop
        self.obs_data = obs_data
        self.loss = loss
        self.epochs = epochs

        self.build_num_batches(n=num_batches, s=batch_size)

        self.build_trainables()
        self.optimizer = optimizer[0](self.trainable, **optimizer[1])
        self.build_scheduler(scheduler)

        self.build_customs(**kw)

    def build_num_batches(self, *, n, s):
        if( n is None ):
            if( s is None ):
                n = 1
            else:
                n = int(np.ceil(self.obs_data.shape[0]/s))
        elif( s is not None ):
            raise ValueError(
                'Only one of num_batches or batch_size can be specified' 
                + f'got num_batches={n}, batch_size={s}'
            )
        self.num_batches = n
    
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

        self.custom['verbosity'] = verbosity
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
    def batch_report_start(self, *, batch): pass

    @abstractmethod
    def batch_report_end(self, *, batch): pass

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
        self.pre_train()
        for epoch in range(self.epochs):
            self.custom['curr_loss'] = 0.0
            self.custom['epoch_start'] = time.time()
            print(f'Custom size = {asizeof(self)}')
            self.pre_step(epoch)
            epoch_loss = 0.0
            self.optimizer.zero_grad()
            src_amp = self.prop.model.survey.src_amp_y()
            batch_size = int(np.ceil(src_amp.shape[0] / self.num_batches))
            v = self.prop.model.vp()
            for batch in range(self.num_batches):
                self.batch_report_start(batch=batch)
                self.custom['batch_start'] = time.time()
                il = batch * batch_size
                ir = min( (batch+1) * batch_size, src_amp.shape[0] )
                if( ir <= il ): 
                    raise ValueError(f'ir={ir}, il={il}, but need ir > il')
                out = dw.scalar(
                    v,
                    self.prop.model.dx,
                    self.prop.model.dt,
                    source_amplitudes=src_amp[il:ir],
                    source_locations=self.prop.model.survey.src_loc_y[il:ir],
                    receiver_locations=self.prop.model.survey.rec_loc_y[il:ir]
                )
                loss = self.loss(out[-1], self.obs_data[il:ir])
                epoch_loss += loss.item()
                loss.backward()
                self.batch_report_end(batch=batch)
            self.custom['epoch_end'] = time.time()
            self.optimizer.step()
            self.scheduler.step()
            self.post_step(epoch)
        self.post_train()

class FWIMetaHandler(FWIAbstract, ABC, metaclass=CombinedMeta):
    rpt: Ant[Callable, 'Report function']
    rpt_prog: Ant[Callable, 'Progress report function']
    rpt_debug: Ant[Callable, 'Debug report function']
    rpt_inf: Ant[Callable, 'Info report function']
    rpt_return: Ant[Callable, 'Return report function']
    closure_calls: Ant[int, 'Number of closure calls']
    take_step_meta: Ant[Callable, 'Take step meta function']

    def __init__(self, **kw):
        super().__init__(**kw)
        self.rpt = self.custom['rpt']
        self.rpt_prog = lambda s: self.rpt(s, idt=1, end='\n', _verbosity_='progress')
        self.rpt_debug = lambda s: self.rpt(s, idt=1, end='\n', _verbosity_='debug')
        self.rpt_inf = lambda s: self.rpt(s, idt=1, end='\n', _verbosity_='inf')
        self.rpt_return = lambda s: self.rpt(s, idt=1, end='\r', _verbosity_='progress')
        self.closure_calls = 0
        self.take_step_meta = self._take_step_meta_()

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
        set_field('loss_scaling', 1.0)
        set_field(
            'log', 
            {
                'loss': [], 
                'grad_norm': {
                    'vp': [],
                    'vs': [],
                    'rho': [],
                    'src_amp_y': [],
                    'src_amp_x': []
                }
            }
        )

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
    
    def batch_report_start(self, *, batch):
        if( batch == 0 ):
            self.rpt_return(f'Starting first batch of {self.num_batches}')
        self.rpt_debug(
            self.debug_data(
                extra_obj=[
                    ('obs_data', self.obs_data)
                ]
            )
        )
        self.rpt_debug(full_mem_report(title=f'Batch {batch}'))
        self.custom['batch_start'] = time.time()

    def batch_report_end(self, *, batch):
        batch_time = time.time() - self.custom['batch_start']
        epoch_time = time.time() - self.custom['epoch_start']
        avg_time_per_batch = epoch_time / (batch+1)
        etr = avg_time_per_batch * (self.num_batches-batch-1)
        self.rpt_return(
            f'Completed {batch+1}/{self.num_batches} -> '
                + f'(batch, total, Epoch ETR) ='
                + f' ({ht(batch_time)}, {ht(epoch_time)}, {ht(etr)})'
        )

    def pre_train(self):
        self.custom['plot_curr'](0)
        self.custom['train_start'] = time.time()
    
    def post_train(self):
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
        self.custom['train_end' ] = time.time()

    def pre_step(self, epoch):
        print_freq = self.custom['print_freq']

        rpt = self.custom['rpt']
        rpt_prog = lambda s: rpt(s, idt=0, end='\n', _verbosity_='progress')
        if( epoch % print_freq == 0 ):
            rpt_prog(f'Epoch {epoch+1}/{self.epochs}')

        self.custom['step_start'] = time.time()

    def post_step(self, epoch): 
        self.custom['plot_curr'](epoch+1)

        # self.rpt_prog(f'Loss: {self.custom["log"]["loss"][-1]:.4e}')
        self.rpt_prog(f'Loss: {self.custom["curr_loss"]:.4e}')
        # for k,v in self.custom['log']['grad_norm'].items():
        #     curr_norm = 'None' if len(v) == 0 or v[-1] is None else f'{v[-1].norm():.4e}'
        #     curr_statement = f'Grad norm "{k}": {curr_norm}'
        #     if( curr_norm == 'None' ):
        #         self.rpt_debug(curr_statement)
        #     else:
        #         self.rpt_prog(curr_statement)

        epoch_time = time.time() - self.custom['step_start']
        total_time = time.time() - self.custom['train_start']
        avg_time_per_epoch = total_time / (epoch+1)
        etr = avg_time_per_epoch * (self.epochs-epoch-1)
        self.rpt_prog(
            f'(Epoch,Total,ETR) = ' 
                + f'({ht(epoch_time)}, {ht(total_time)}, {ht(etr)})'
        )
        self.custom['step_end'] = time.time()

    @abstractmethod 
    def _take_step_(self, *, epoch, batch, idx, **kw): pass

    def _take_step_meta_(self):
        if( self.custom['verbosity'] == 'silent' ):
            return self._take_step_
        else:
            def helper(*, epoch, batch, idx, **kw):
                batch_start = self.batch_report_start(batch=batch)
                loss_lcl = self._take_step_(
                    epoch=epoch, 
                    batch=batch, 
                    idx=idx, 
                    **kw
                )
                self.batch_report_end(
                    epoch=epoch,
                    epoch_start=self.custom['step_start'],
                    epoch_loss=loss_lcl,
                    batch=batch,
                    batch_idx=idx,
                    batch_start=batch_start
                )
                return loss_lcl
            return helper
    
    def take_step(self, *, epoch, batch, idx, **kw):
        return self.take_step_meta(epoch=epoch, batch=batch, idx=idx, **kw)

class FWI(FWIMetaHandler, metaclass=SlotMeta):
    def postprocess_loss(self):
        if( 'clip_grad' in self.custom.keys() ):
            for att, clip_val in self.custom['clip_grad']:
                a = getattr(self.prop.model, att).param
                torch.nn.utils.clip_grad_value_(
                    a,
                    torch.quantile(a.grad.detach().abs(), 0.98)
                )
    
    def _take_step_(self, *, epoch, batch, idx, **kw):
        #gpu_mem('Enter step', color=(5,3,5))
        # self.optimizer.zero_grad()
        #gpu_mem('Before forward', color=(5, 0, 0))
        out = self.prop(
            idx=idx,
            device=torch.device('cuda'),
            **self.custom['forward_kwargs']
        )
        #gpu_mem('After forward', color=(0, 5, 0))
        loss = self.custom['loss_scaling'] \
            * self.loss(out, self.obs_data[idx])
        loss.backward()
        #gpu_mem('After backward', color=(0, 0, 5))
        self.custom['curr_loss'] += loss.item()
        self.postprocess_loss()
    
        # self.optimizer.step()
        # self.scheduler.step()
        #gpu_mem('After step', color=(5, 5, 0))

        # del loss, out
        # torch.cuda.empty_cache()
        #gpu_mem('After empty cache', color=(5, 0, 5))
        # return {'batch_loss': batch_loss}