from subprocess import check_output as co
from subprocess import CalledProcessError
import sys
from time import time
import matplotlib.pyplot as plt
from imageio import imread, mimsave
import numpy as np
import torch
from typing import Annotated as Ant, Any, Optional as Opt, Callable
from abc import ABCMeta, abstractmethod
import itertools
from .swiffer import *
from torch.optim.lr_scheduler import _LRScheduler
import deepwave as dw
from warnings import warn

def gpu_mem(msg='', color='red', print_protocol=print):
    if( len(msg) > 0 and msg[-1] != '\n' ): msg += '\n'

    if( type(color) == tuple ):
        color = [str(e) for e in color]
        color = 'rgb' + '_'.join(color)
    out = sco_bash('gpu_mem', color, split=True)
    out = [f'    {e}' for e in out if len(e) > 0]
    out[-1] = out[-1].replace('\n', '')
    out = '\n'.join(out)
    print_protocol(f'{msg}{out}')

def gaussian_perturb(ref, scaled_sigma, scaled_mu, scale=False):
    if( scale ):
        scaling = torch.max(torch.abs(ref))
    else:
        scaling = 1.0
    sigma = scaled_sigma * scaling
    mu = scaled_mu * scaling
    noise = torch.randn_like(ref) * sigma + mu
    tmp = ref + noise
    v = tmp.clone().requires_grad_()
    return v

def verbosity_str_to_int(*, verbosity, levels):
    if( type(verbosity) == int ): return verbosity
    elif( type(verbosity) == str ):
        verbosity = verbosity.lower()
        for (level, level_names) in levels:
            if( verbosity in level_names ): return level
        raise ValueError(f'Verbose value {verbosity} not recognized')
    else:
        raise ValueError(
            f'Verbosity must be int or str, got {type(verbosity)}'
        )

def clean_levels(levels):  
    if( levels is None ):
        levels = []
        levels.append((0, ['none', 'silent']))
        levels.append((1, ['low', 'progress']))
        levels.append((2, ['medium', 'debug']))
        levels.append((np.inf, ['high', 'all']))
    for (i,l) in enumerate(levels):
        if( type(l) is int ):
            levels[i] = (l, [str(l)])
    for (i,l) in enumerate(levels):
        if( type(l) not in [list, tuple] or len(l) != 2 ):
            raise ValueError('Levels must be list of pairs')
        elif( type(l[1]) is not list ):
            raise ValueError(f'Level names must be list, got {type(l[1])}')
        elif( str(l[0]) not in l[1] ):
            l[1].append(str(l[0]))
            if( l[0] is np.inf ):
                l[1].append('infinity')
    levels = sorted(levels, key=lambda x: x[0])
    return levels

def run_verbosity(*, verbosity, levels):
    levels = clean_levels(levels)  
    v2i = lambda x: verbosity_str_to_int(verbosity=x, levels=levels)
    verbosity_int = v2i(verbosity)
    def helper(f):
        def helper_inner(*args, _verbosity_, **kw):
            _verbosity_int = v2i(_verbosity_)
            if( _verbosity_int <= verbosity_int ):
                return f(*args, **kw)
            else:
                return None
        return helper_inner
    return helper

def mem_report(*args, precision=2, sep=', ', rep=None):
    filtered_args = []
    if( rep is None ):
        rep = []
    [rep.append('unknown') for _ in range(len(args) - len(rep))]
    add = lambda x, i: filtered_args.append(x + ' (' + rep[i] + ')')
    for (i,arg) in enumerate(args):
        if( 1e18 < arg ):
            add(f'{arg/1e18:.{precision}f} EB', i)
        elif( 1e15 < arg ):
            add(f'{arg/1e15:.{precision}f} PB', i)
        elif( 1e12 < arg ):
            add(f'{arg/1e12:.{precision}f} TB', i)
        elif( 1e9 < arg ):
            add(f'{arg/1e9:.{precision}f} GB', i)
        elif( 1e6 < arg ):
            add(f'{arg/1e6:.{precision}f} MB', i)
        elif( 1e3 < arg ):
            add(f'{arg/1e3:.{precision}f} KB', i)
        else:
            add(f'{arg:.{precision}f} B', i)
    return sep.join(filtered_args)

def full_mem_report(precision=2, sep=', ', rep=('free', 'total'), title=None):
    if( title is None ): title = ''
    else: title = title + '\n    '
    return title \
        + mem_report(
            *torch.cuda.mem_get_info(), 
            precision=precision, 
            sep=sep, 
            rep=rep
        )

def taper(x, length):
    return dw.common.cosine_taper_end(x, length)

class SlotMeta(type):
    def __new__(cls, name, bases, class_dict):
        # Extract the variable names from the annotations
        try:
            annotated_keys = list(
                class_dict['__annotations__'].keys()
            )
        except KeyError:
            annotated_keys = []
        
        # Find attributes that are not methods, not in special names and not already annotated
        non_annotated_attrs = [
            key for key, value in class_dict.items() 
                if not (
                    callable(value) 
                    or key.startswith('__') 
                    or key in annotated_keys
                )
        ]
        
        # Add the default annotations for non-annotated attributes
        for key in non_annotated_attrs:
            class_dict['__annotations__'][key] = Ant[Any, 'NOT ANNOTATED']
            
            # Optional: Remove the attributes as they'll be defined by __slots__ 
            class_dict.pop(key, None)

        # Create the __slots__ attribute from updated annotationsi
        try:
            class_dict['__slots__'] = list(
                class_dict['__annotations__'].keys()
            )
        except KeyError:
            class_dict['__slots__'] = []
                
        return super().__new__(cls, name, bases, class_dict)
    
class CombinedMeta(SlotMeta, ABCMeta):
    pass

class AbstractParam(torch.nn.Module, metaclass=CombinedMeta):
    param: Ant[torch.nn.Parameter, 'Parameter']

    def __init__(self, *, param):
        super().__init__()
        self.param = param

    @abstractmethod
    def forward(self, **kw):
        raise NotImplementedError('Forward not implemented')
    
class Param(AbstractParam):
    def forward(self):
        return self.param
    
class ConstrainedParam(AbstractParam):
    def __init__(
        self, 
        *, 
        param, 
        trainable, 
        device='cpu', 
        min_val, 
        max_val
    ):
        param = torch.logit((param - min_val) / (max_val - min_val))
        super().__init__(
            param=param,
            trainable=trainable,
            device=device,
            min_val=min_val,
            max_val=max_val
        )
    
    def forward(self, *, idx='all'):
        if( idx == 'all' ):
            return torch.sigmoid(self.param) \
                * (self.max_val - self.min_val) \
                + self.min_val
        else:
            return torch.sigmoid(self.param[idx]) \
                * (self.max_val - self.min_val) \
                + self.min_val

class ParamFWI(torch.nn.Module, metaclass=SlotMeta):
    param: Ant[torch.nn.Parameter, 'Parameter']
    forward: Ant[Callable, 'Parameter']
    custom: Ant[dict, 'Custom metadata']

    def __init__(
        self, 
        *, 
        initial, 
        setup=None, 
        forward=None, 
        requires_grad=False,
        store_kw=False,
        **kw
    ):
        if( initial is None ):
            self.param = None
            return

        super().__init__()
        setup, self.forward = self.builder(setup=setup, forward=forward, **kw)
        self.param = torch.nn.Parameter(setup(initial))
        self.param.requires_grad = requires_grad

        if( store_kw ): self.custom = kw

    def builder(self, *, setup, forward, **kw):
        pre_setups, extract = self.setup_predefinitions()
        setup_key = 'user_defined_callables'
        for k, v in pre_setups.items():
            if( setup in v[0] and forward in v[1] ):
                setup_key = k
                break
            elif( setup in v[0] ):
                raise ValueError(
                    f'If setup is in {v[0]}, forward must be in {v[1]}, ' +
                    f'got setup={setup}, forward={forward}'
                )
        return extract(setup=setup, forward=forward, key=setup_key, **kw)

    def setup_predefinitions(self):
        pre_setups = {
            'identity': ((None, 'identity'), (None, 'identity')),
            'logit': (
                (None, 'logit', 'constrained'), 
                ('sigmoid', 'logit_inverse', 'constrained')
            )
        }
        def extract(*, setup, forward, key, **kw):
            if( key == 'identity' ):
                return lambda x: x, lambda x: x
            elif( key == 'logit' ):
                try: 
                    minv, maxv = kw['min_val'], kw['max_val']
                except KeyError:
                    raise ValueError(
                        f'Setup=={setup} requires min_val and max_val ' +
                        f'to be specified in kw, got {kw.keys()}'
                    )
                def logit(x):
                    return torch.logit( (x-minv) / (maxv-minv) )
                def logit_inv(x):
                    return torch.sigmoid(x) * (maxv-minv) + minv
                return logit, logit_inv
            elif( key == 'user_defined_callables' ):
                if( not callable(setup) or not callable(forward) ):
                    raise ValueError(
                        f'If setup and forward are not in predefined_setups, ' +
                        f'they must be callable, got ' +
                        f'type(setup)={type(setup)}, ' +
                        f'type(forward)={type(forward)}'
                    )
                return setup, forward
            else:
                raise ValueError(f'BUG: Unexpected setup_key={key}')
        return pre_setups, extract

class WaveModel(torch.nn.Module, metaclass=SlotMeta):
    vp: Ant[ParamFWI, 'ParamFWI']
    vs: Opt[Ant[ParamFWI, 'ParamFWI']]
    rho: Opt[Ant[ParamFWI, 'ParamFWI']]
    src_amp_y: Ant[ParamFWI, 'ParamFWI']
    src_amp_x: Opt[Ant[ParamFWI, 'ParamFWI']]

    def __init__(self, *, vp, src_amp_y, vs=None, rho=None, src_amp_x=None):
        super().__init__()
        self.vp = vp
        self.src_amp_y = src_amp_y
        self.vs = vs
        self.rho = rho
        self.src_amp_x = src_amp_x
    
    def forward(
        self, 
        *, 
        dx,
        dt,
        src_loc_y, 
        rec_loc_y, 
        model='acoustic',
        **kw
    ):
        if( model == 'acoustic' ):
            return dw.acoustic(
                self.vp(),
                dx,
                dt,
                source_amplitudes=self.src_amp_y(),
                source_locations=src_loc_y,
                receiver_locations=rec_loc_y,
                **kw
            )[-1]
        elif( model == 'elastic' ):
            return dw.elastic(
                vp=self.vp(),
                vs=self.vs(),
                rho=self.rho(),
                dx=dx,
                dt=dt,
                source_amplitudes_y=self.src_amp_y(),
                source_locations_y=src_loc_y,
                receiver_locations_y=rec_loc_y,
                **kw
            )[-2]
        else:
            raise ValueError(f'Unknown model {model}')
        
class DotDict:
    def __init__(self, d):
        for k,v in d.items():
            setattr(self, k, v)