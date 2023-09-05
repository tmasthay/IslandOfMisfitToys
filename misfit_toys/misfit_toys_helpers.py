import os
from subprocess import check_output as co
from subprocess import CalledProcessError
import sys
from time import time
import matplotlib.pyplot as plt
from imageio import imread, mimsave
import numpy as np
import torch
from typing import Annotated as Ant, Any
from abc import ABCMeta
import itertools
from .base_helpers import *
from .misfit_toys_helpers_helpers.download_data import *
from torch.optim.lr_scheduler import _LRScheduler
import deepwave as dw

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

def run_and_time(start_msg, end_msg, f, *args, **kwargs):
    stars = 80*'*' + '\n'
    print('%s\n%s'%(stars, start_msg), file=sys.stderr)
    start_time = time()
    u = f(*args, **kwargs)
    print('%s ::: %.4f\n%s'%(end_msg, time() - start_time, stars), 
        file=sys.stderr)
    return u

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

def gpu_mem_helper():
    with open(os.path.expanduser('~/.bash_functions'), 'r') as f:
        lines = f.readlines()

    start_line = next(i for i, line in enumerate(lines) if line.strip() == 'gpu_mem() {')
    end_line = next(i for i, line in enumerate(lines) if i > start_line and line.strip() == '}')

    s = ''.join(lines[start_line+1:end_line])
    def helper(msg=''):
        print('%s...%s'%(msg, ':::'.join(sco(s))))
    return helper

def add_bullseye(ax, x, y, s, color_seq, alphas, **kw):
    def listify(x):
        if( type(x) != list ): return 3 * [x]
        elif( len(x) == 1 ): return 3 * [x[0]]
        else: return x
    color_seq = listify(color_seq)
    alphas = listify(alphas)
    assert( len(alphas) == len(color_seq) )
    L = len(color_seq)
    for (i,c) in enumerate(color_seq):
        ax.add_patch(
            plt.Circle(
                (x,y), 
                s*(L-i)/L, 
                color=c, 
                alpha=alphas[i],
                **kw
            )
        )
        plt.scatter(x,y,s=0.0, color='w')

def constant_array(x):
    return np.all([torch.all(e == x[0]) for e in x])

def get_survey_type(src, rec):
    if( constant_array(rec) ):
        return 'Common Gather'
    elif( constant_array( rec.unsqueeze(2) - src.unsqueeze(1) ) ):
        return 'Common Offset'
    elif( constant_array( rec.unsqueeze(2) + src.unsqueeze(1) ) ):
        return 'Common Midpoint'
    else:
        return 'Irregular Survey'
    
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

def vertical_stratify(ny, nx, layers, values, device):
    assert( len(layers) == len(values) - 1 )
    assert( len(values) >= 1 )
    u = values[0] * torch.ones(ny,nx, device=device)
    if( len(layers) > 0 ):
        for l in range(len(layers)-1):
            u[layers[l]:layers[l+1], :] = values[l+1]
        u[layers[-1]:] = values[-1]
    return u

def uniform_vertical_stratify(ny, nx, values, device):
    layers = [i*ny // len(values) for i in range(1,len(values))]
    return vertical_stratify(ny, nx, layers, values, device)

def plot_material_params(vp, vs, rho, cmap):
    up = vp.cpu().detach()
    us = vs.cpu().detach()
    urho = rho.cpu().detach()
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(10,7))

    alpha = 0.3
    im0 = axs[0].imshow(up, cmap=cmap)
    axs[0].set_title(r'$V_p$')
    fig.colorbar(im0, ax=axs[0],shrink=alpha)

    im1 = axs[1].imshow(us, cmap=cmap)
    axs[1].set_title(r'$V_s$')
    fig.colorbar(im1, ax=axs[1], shrink=alpha)

    im2 = axs[2].imshow(urho, cmap=cmap)
    axs[2].set_title(r'$\rho$')
    fig.colorbar(im2, ax=axs[2], shrink=alpha)

    for i in range(3):
        axs[i].set_xlabel('Horizontal location (km)')
        axs[i].set_ylabel('Depth (km)')

    plt.subplots_adjust(wspace=0.5, hspace=0.5)

    plt.savefig('params.pdf')
    plt.clf()

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
    res = torch.zeros(n_shots, src_per_shot, 2)
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

def open_ide(*args, ide_precedence=True, no_ide=[], default='/usr/bin/open'):
    cmd = default
    def scan_no_ide():
        nonlocal cmd
        for open_cmd in no_ide:
            check1 = sco('which %s'%open_cmd)
            check2 = sco('type %s'%open_cmd)
            if( bool(check1 or check2) ):
                cmd = open_cmd
                break
    if( not ide_precedence ): scan_no_ide()
    if( cmd == default ):
        python_parent_pid = os.getppid()
        shell_parent = sco(
            f'ps -p $(ps -o ppid= -p {python_parent_pid}) -o comm='
        )[0] \
        .strip()
        cmd = default
        for a in args:
            ide, open_cmd = a[0].lower(), a[1]
            if( shell_parent == ide ):
                cmd = open_cmd
                break
    if( ide_precedence and cmd == default ): scan_no_ide()
    def helper(file_name):
        os.system(f'{cmd} {file_name}')
    return helper

def fetch_and_convert_data(
    *,
    subset='all',
    path=os.getcwd(),
    check=True
):
    datasets = {
        'marmousi': {
            'url': 'https://www.geoazur.fr/WIND/pub/nfs/FWI-DATA/' + 
                'GEOMODELS/Marmousi',
            'ext': 'bin',
            'ny': 2301,
            'nx': 751,
            'vp': {},
            'rho': {}
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

    fetch_data(datasets, path=path)
    convert_data(datasets, path=path)

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

def retrieve_dataset(
    *, 
    field, 
    folder, 
    path=os.getcwd(),
    check=False
):

    if( path is None ): 
        path = ''
    if( path == 'conda' ):
        path = os.path.join(sco('echo $CONDA_PREFIX')[0], 'data')
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
    
    fetch_and_convert_data(subset=folder, path=path, check=check)
    return torch.load(os.path.join(full_path, f'{field}.pt'))

def device_deploy(obj, deploy):
    if len(deploy) > 0 and deploy[0][0] == 'all':
        device = deploy[0][1]
        for field in dir(obj):
            if not field.startswith("__"):
                curr = getattr(obj, field)
                if isinstance(curr, torch.Tensor):
                    setattr(obj, field, curr.to(device))
    else:
        for att, device in deploy:
            curr = getattr(obj, att)
            if curr is None:
                raise ValueError(f'Attribute {att} in {obj} is None')
            setattr(obj, att, curr.to(device))

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

def get_member_name(obj, sub_obj, special=False):
    if( special ):
        l = dir(obj)
    else:
        l = [e for e in dir(obj) if not e.startswith('__')]
    fields = [i for i,e in enumerate(l) if id(getattr(obj, e)) == id(sub_obj)]

    if( len(fields) == 0 ):
        raise ValueError(f'{sub_obj} not found in {obj}')
    elif( len(fields) > 1 ):
        raise ValueError(f'{sub_obj} found multiple times in {obj}')
    else:
        return l[fields[0]]

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

class ConstrainedVelocity(torch.nn.Module):
    def __init__(self, *, initial, min_vel, max_vel):
        super().__init__()
        self.min_vel = min_vel
        self.max_vel = max_vel
        self.model = torch.nn.Parameter(
            torch.logit((initial - min_vel) / (max_vel - min_vel))
        )

    def forward(self):
        return torch.sigmoid(self.model) * (self.max_vel - self.min_vel) \
            + self.min_vel
    
class IdentityVelocity(torch.nn.Module):
    def __init__(self, *, initial):
        super().__init__()
        self.model = torch.nn.Parameter(initial)

    def forward(self):
        return self.model

class Prop(torch.nn.Module):
    def __init__(self, *, v, dx, dt, freq, deploy=[]):
        super().__init__()
        self.v = v
        self.dx = dx
        self.dt = dt
        self.freq = freq
        device_deploy(self, deploy)

    def forward(
        self,
        *,
        source_amplitudes, 
        source_locations, 
        receiver_locations
    ):
        max_vel = torch.max(self.v.model)
        return dw.scalar(
            self.v.model,
            self.dx, 
            self.dt,
            source_amplitudes=source_amplitudes,
            source_locations=source_locations,
            receiver_locations=receiver_locations,
            max_vel=max_vel,
            pml_freq=self.freq,
            time_pad_frac=0.2,
        )
    
def build_prop(
    *,
    vp,
    dx,
    dt,
    freq,
    device,
    multi_gpu=True
):
    v = IdentityVelocity(initial=vp.to(device))
    prop = Prop(v=v, dx=dx, dt=dt, freq=freq)
    device_deploy(prop, [('all', device)])
    if( multi_gpu ):
        prop = torch.nn.DataParallel(prop).to(device)
    return prop

    
