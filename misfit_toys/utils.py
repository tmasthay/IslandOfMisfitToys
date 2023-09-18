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
from .basement.download_data import *
from torch.optim.lr_scheduler import _LRScheduler
import deepwave as dw
from warnings import warn

def get_file(s, path=''):
    full_path = list(set(path.split(':')) \
        .union(set(sco('echo $CONDA_PREFIX/data'))))
    full_path = [e for e in full_path if e != '']
    if( os.getcwd() not in full_path ):
        full_path.insert(0, os.getcwd())
    for e in full_path:
        if( os.path.exists(e + '/' + s) ): return e + '/' + s
    stars = 80 * '*'
    raise FileNotFoundError('filename "%s" not found in any' + \
        ' of the following directories\n%s\n%s\n%s'%(
            s, stars, '\n'.join(full_path), stars
        )
    )

def make_gif(x, folder, the_map='cividis'):
    os.system('mkdir -p %s'%folder)
    for i in range(len(x)):
        plt.imshow(np.transpose(x[i]), cmap=the_map, aspect='auto')
        plt.colorbar()
        plt.title('Epoch %d'%i)
        plt.savefig('%s/%d.jpg'%(folder, i))
        plt.close()
    filenames = ['%s/%d.jpg'%(folder, i) for i in range(len(x))]
    images = [imread(e) for e in filenames]
    mimsave('%s/movie.gif'%folder, images, duration=0.1, loop=0)
    for e in filenames:
        print(e)
        os.system('rm %s'%e)

def report_gpu_memory_allocation(msg, mode=2):
    memory_gb = torch.cuda.memory_allocated() / (1024 ** 3)
    print(f"GPU Memory {msg}: {memory_gb:.2f} GB")

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

def read_tensor(s, device):
    if( s == None ): return None
    elif( type(s) == str ): return torch.load(s, device=device)
    else: return s.to(device)

def towed_src(
    *,
    n_shots,
    src_per_shot,
    fst_src,
    d_src,
    src_depth,
    d_intra_shot
):
    res = torch.zeros(n_shots, src_per_shot, 2, dtype=torch.long)
    res[:, :, 1] = src_depth
    for i in range(n_shots):
        for j in range(src_per_shot):
            res[i, j, 0] = fst_src + i * d_src + j * d_intra_shot
    return res

def fixed_rec(
    *,
    n_shots,
    rec_per_shot,
    fst_rec,
    d_rec,
    rec_depth
):
    res = torch.zeros(n_shots, rec_per_shot, 2)
    res[:, :, 1] = rec_depth
    res[:, :, 0] = (torch.arange(rec_per_shot) * d_rec + fst_rec) \
        .repeat(n_shots, 1)
    return res  

def get_all_devices():
    gpus = [
        torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())
    ]
    return gpus + [torch.device('cpu')]

def fetch_and_convert_data(
    *,
    subset='all',
    path=os.getcwd(),
    check=False
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    datasets = {
        'marmousi': {
            'url': 'https://www.geoazur.fr/WIND/pub/nfs/FWI-DATA/' + 
                'GEOMODELS/Marmousi',
            'ext': 'bin',
            'ny': 2301,
            'nx': 751,
            'vp': {},
            'rho': {},
            'obs_data': (create_obs_marm_dw, (), {'device': device})
        },
        'marmousi2': {
            'url': 'http://www.agl.uh.edu/downloads/',
            'ext': 'segy',
            'vp': {'filename': 'vp_marmousi-ii.segy.gz'},
            'vs': {'filename': 'vs_marmousi-ii.segy.gz'},
            'rho': {'filename': 'density_marmousi-ii.segy.gz'}
        },
        'DAS': {
            'url': 'https://ddfe.curtin.edu.au/7h0e-d392/',
            'ext': 'sgy',
            'das_curtin': {'filename': '2020_GeoLab_WVSP_DAS_wgm.sgy'},
            'geophone_curtin': {
                'filename': '2020_GeoLab_WVSP_geophone_wgm.sgy'
            },
        }
    }
    datasets = expand_metadata(datasets)
   
    if( type(subset) == str ):
        subset = [e.strip() for e in subset.split(' ')]

    if( path == '' or '/' != path[0] ):
        path = os.path.join(os.getcwd(), path)

    if( 'all' not in subset 
       and set(subset) != set(datasets.keys()) 
    ):
        datasets = {k:v for k,v in datasets.items() if k in subset} 

    calls = fetch_data(datasets, path=path)
    convert_data(datasets, path=path, calls=calls)
    store_metadata(metadata=datasets, path=path)

    if( check ):
        res = check_data_installation(path)
        if( res is None ):
            print('NO PYTORCH FILES FOUND')
        else:
            total = len(res['success']) + len(res['failure'])
            success_head = 'SUCCESS: %d / %d'%(len(res['success']), total)
            print(f'\n{success_head}\n' + '*'*len(success_head))
            print('\n'.join(res['success']))

            failure_head = 'FAILURE: %d / %d'%(len(res['failure']), total)     
            print(f'\n{failure_head}\n' + '*'*len(failure_head))
            print('\n'.join(res['failure']))
        
    return datasets

def get_data(
    *, 
    field, 
    folder, 
    path=None,
    check=False
):

    if( path in [None, 'conda'] ): 
        path = os.path.join(sco('echo $CONDA_PREFIX')[0], 'data')
    elif( path == 'pwd' ):
        path = os.getcwd()
    elif( path == '' or path[0] != '/' ):
        path = os.path.join(os.getcwd(), path)
    
    full_path = os.path.join(path, folder)
    if( os.path.exists(full_path) ):
        try:
            return torch.load(os.path.join(full_path, f'{field}.pt'))
        except FileNotFoundError:
            print(
                f'File {field}.pt not found in {full_path}' +
                f'\n    Delete {folder} in {path} and try again'
            )
            raise
    # add another 
    fetch_and_convert_data(subset=folder, path=path, check=check)
    return torch.load(os.path.join(full_path, f'{field}.pt'))

def get_data2(*, field, path=None):
    path_tok = path.split('/')
    if( path_tok[0] in [None, 'conda'] ):
        path_tok[0] = os.environ['CONDA_PREFIX'] + '/data'
    elif( path_tok[0] == 'pwd' ):
        path_tok[0] = os.getcwd()
    elif( path_tok[0] != '' ):
        path_tok[0] = os.path.join(os.getcwd(), path_tok[0])
    path = '/'.join(path_tok)
    field_file = os.path.join(path, f'{field}.pt')
    if( os.path.exists(path) ):
        try:
            return torch.load(field_file)
        except FileNotFoundError:
            print(
                f'File {field}.pt not found in {path}' +
                f'\n    Delete {path} and try again'
            )
            raise
    fetch_and_convert_data(subset=path_tok[-1], path='/'.join(path_tok[:-1]))
    return torch.load(field_file)

def sub_dict(d, keys):
    return {k:v for k,v in d.items() if k in keys}

def set_nonslotted_params(obj, params):
    keys = [k for k in params.keys() if k not in obj.__slots__]
    if( not hasattr(obj, 'custom') ):
        obj.custom = {}
    obj.custom.update(sub_dict(params, keys))

def downsample_tensor(tensor, axis, ratio):
    """
    Downsample a torch.Tensor along a given axis by a specific ratio.
    
    Parameters:
        tensor (torch.Tensor): The input tensor to downsample.
        axis (int): The axis along which to downsample. Must be in range [0, tensor.dim()).
        ratio (int): The downsampling ratio. Must be greater than 0.
        
    Returns:
        torch.Tensor: The downsampled tensor.
    """
    
    if ratio <= 0:
        raise ValueError("Ratio must be greater than 0")
        
    if axis < 0 or axis >= tensor.dim():
        raise ValueError(f"Axis must be in range [0, {tensor.dim()}).")
        
    slices = [slice(None)] * tensor.dim()
    slices[axis] = slice(None, None, ratio)
    
    return tensor[tuple(slices)]

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

def create_obs_marm_dw(path, device, overwrite=False):
    if( path == 'conda' ):
        path = os.path.join(sco('echo $CONDA_PREFIX')[0], 'data')
    warn(
        '\n    This function contains a CUDA memory leak!' 
        '\n    Run this prior to your main script and then use ' 
        '\n    get_data function to pull in obs_data.'
        '\n    Doing so will make this small memory leak only occur once'
        ' and be cleaned up by the garbage collector and thus benign.'
    )
    try:
        print('Attempt obs_data fetch...', end='')
        obs_data = get_data(
            field='obs_data', 
            folder='marmousi', 
            path=path
        )
        print(f'Found marmousi data in {path}, shape={obs_data.shape}')
        if( not overwrite ):
            print(
                'Not overwriting, returning obs_data...'
                'set overwrite=True to overwrite'
            )
            return obs_data
        print('OVERWRITE WARNING! CTRL+C OR KILL NOW IF YOU WANT TO KEEP IT')
        print('Sleeping for 3 seconds...', end='')
        time.sleep(3)
        print(
            f'Proceeding to overwrite ' +
            f'{os.path.join(path, "marmousi/obs_data")}'
        )
        del obs_data
    except FileNotFoundError:
        print(f'No marmousi observation data in {path}, creating now...')


    vp = get_data(field='vp', folder='marmousi', path=path).to(device)
    n_shots = 115

    n_sources_per_shot = 1
    d_source = 20  # 20 * 4m = 80m
    first_source = 10  # 10 * 4m = 40m
    source_depth = 2  # 2 * 4m = 8m

    n_receivers_per_shot = 384
    d_receiver = 6  # 6 * 4m = 24m
    first_receiver = 0  # 0 * 4m = 0m
    receiver_depth = 2  # 2 * 4m = 8m

    freq = 25
    nt = 750
    dt = 0.004
    peak_time = 1.5 / freq

    source_locations = torch.zeros(n_shots, n_sources_per_shot, 2,
                                   dtype=torch.long).to(device)
    source_locations[..., 1] = source_depth
    source_locations[:, 0, 0] = (torch.arange(n_shots) * d_source +
                                 first_source)

    # receiver_locations
    receiver_locations = torch.zeros(n_shots, n_receivers_per_shot,
                                     2, dtype=torch.long).to(device)
    receiver_locations[..., 1] = receiver_depth
    receiver_locations[:, :, 0] = (
        (torch.arange(n_receivers_per_shot) * d_receiver +
         first_receiver)
        .repeat(n_shots, 1)
    )

    # source_amplitudes
    source_amplitudes = (
        (dw.wavelets.ricker(freq, nt, dt, peak_time))
        .repeat(n_shots, n_sources_per_shot, 1)
    ).to(device)

    dx = 4.0
    out = dw.scalar(
        vp,
        dx,
        dt,
        source_amplitudes=source_amplitudes,
        source_locations=source_locations,
        receiver_locations=receiver_locations,
        pml_freq=freq,
        accuracy=8
    )[-1]
    out_cpu = out.to('cpu')

    torch.save(out_cpu, os.path.join(path, 'marmousi/obs_data.pt'))
    del source_amplitudes, source_locations, receiver_locations, vp
    del out, out_cpu
    torch.cuda.empty_cache()

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