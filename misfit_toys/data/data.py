import torch
import os
import numpy as np
from obspy.io.segy.segy import _read_segy
import torch
from subprocess import check_output as co
import os
import time
import sys
import torch
from ..swiffer import sco
import re
from warnings import warn
import deepwave as dw
from abc import ABC, abstractmethod

def auto_path(make_dir=False):
    def decorator(func):
        def wrapper(*args, **kwargs):
            if 'path' in kwargs:
                kwargs['path'] = parse_path(kwargs['path'])
                if make_dir:
                    os.makedirs(kwargs['path'], exist_ok=True) 
            return func(*args, **kwargs)
        return wrapper
    return decorator

def expand_metadata(meta):
    d = dict()
    for folder,folder_meta in meta.items():
        d[folder] = dict()
        base = {k:v for k,v in folder_meta.items() if type(v) != dict}
        files = {k:v for k,v in folder_meta.items() if type(v) == dict}
        for filename, file_meta in files.items():
            d[folder][filename] = {**base, **file_meta}
            if( not d[folder][filename]['url'].endswith('/') ):
                d[folder][filename]['url'] += '/'
            if 'filename' not in d[folder][filename].keys():
                d[folder][filename]['filename'] = \
                    filename + '.' + d[folder][filename]['ext']
    return d

def segy_to_torch(
    *,
    input_path, 
    output_path,
    device='cpu', 
    transpose=False, 
    out=sys.stdout,
    print_freq=100,
    **kw
):
    print('READING in SEGY file "%s"'%input_path, file=out)

    # Read the SEGY file
    stream = _read_segy(input_path)

    print('DONE reading SEGY file "%s"'%input_path, file=out)

    num_traces,trace_length = len(stream.traces), len(stream.traces[0].data)
    data_array = np.empty((num_traces, trace_length))

    t = time.time()
    # Loop through each trace in the stream
    for (i,trace) in enumerate(stream.traces):
        if( i == 0 ): print('READING first trace', file=out)
        # Add the trace's data to the list
        data_array[i] = trace.data

        if( i % print_freq == 0 and i > 0 ):
            elapsed = time.time() - t
            avg = elapsed / i
            rem = (num_traces-i) * avg
            print(
                'Elapsed (s): %f, Remaining estimate (s): %f'%(elapsed, rem), 
                file=out
            )

    print('Converting list to pytorch tensor (may take a while)')
    conv = torch.Tensor(data_array).to(device) if not transpose else \
        torch.Tensor(data_array).transpose(0,1).to(device)
    torch.save(conv, output_path)

def bin_to_torch(
    *,
    input_path,
    output_path, 
    device='cpu', 
    transpose=False, 
    out=sys.stdout,
    ny,
    nx,
    **kw
):
    u = torch.from_file(input_path, size=ny*nx)
    torch.save(u.reshape(ny,nx).to(device), output_path)

def any_to_torch(
    *,
    input_path,
    output_path,
    device='cpu',
    transpose=False,
    out=sys.stdout,
    **kw
):
    if( input_path.endswith('.segy') or input_path.endswith('.sgy') ):
        segy_to_torch(
            input_path=input_path, 
            output_path=output_path,
            device=device, 
            transpose=transpose, 
            out=out,
            **kw
        )
    elif( input_path.endswith('.bin') ):
        bin_to_torch(
            input_path=input_path,
            output_path=output_path,
            device=device,
            transpose=transpose,
            out=out,
            **kw
        )
    else:
        raise ValueError(f'Unknown file type: {input_path}')
     
def fetch_data(d, *, path, unzip=True):
    convert_search = dict()
    calls = []
    for folder, info in d.items():
        convert_search[folder] = []

        # make folder if it doesn't exist
        curr_path = os.path.join(path, folder)
        os.makedirs(curr_path, exist_ok=True)
        
        for file, meta in info.items():
            url = os.path.join(meta['url'], meta['filename'])
            # file_path = f'{folder}/{meta["filename"]}'
            file_path = os.path.join(curr_path, meta['filename'])
            print(f'ATTEMPT: {url} -> {file_path}') 
            os.system(f'curl {url} --output {file_path}') 
            if( unzip and meta['filename'].endswith('.gz') ):
                os.system(f'gunzip {file_path}')
                d[folder][file]['filename'] = \
                    d[folder][file]['filename'].replace('.gz', '')
            for k,v in meta.items():
                if( type(v) == tuple ):
                    func = v[0]
                    args = v[1]
                    kwargs = v[2]
                    clos = lambda: func(*args, path=path, **kwargs)
                    calls.append(clos)
    return calls

def convert_data(d, *, path, calls=None):
    for folder, files in d.items():
        for field, meta in files.items():
            curr = os.path.join(path, folder)
            any_to_torch(
                input_path=(
                    os.path.join(curr, meta['filename'])
                ),
                output_path=os.path.join(curr, f'{field}.pt'),
                **meta
            )
        os.system('rm %s/*.%s'%(
                curr,
                f' {curr}/*.'.join(['bin', 'sgy', 'segy', 'gz'])
            )
        )
    for call in calls:
        call()

def check_data_installation(path):
    pytorch_files = sco(f'find {path} -name "*.pt"')
    res = {'success': [], 'failure': []}
    if( pytorch_files is None or len(pytorch_files) == 0 ):
        print('NO PYTORCH FILES FOUND')
        return None
    
    for file in pytorch_files:
        try:
            u = torch.load(file)
            print(f'SUCCESS "{file}" shape={u.shape}')
            res['success'].append(file)
        except:
            print(f'FAILURE "{file}"')
            res['failure'].append(file)
    return res

def prettify_dict(d, jsonify=True):
    s = str(d)
    s = re.sub(r'<function (\w+) at 0x[\da-f]+>', r'\1', s)
    s = s.replace('{', '{\n')
    s = s.replace('}', '\n}')
    s = s.replace(', ', ',\n')
    lines = s.split('\n')
    idt = 4*' '
    idt_level = 0
    for (i,l) in enumerate(lines):
        if( l in ['}', '},', ','] ):
            idt_level -= 1
            if( idt_level < 0 ):
                idt_level = 0
        lines[i] = idt_level*idt + l
        if( l[-1] == '{' ):
            idt_level += 1
    res = '\n'.join(lines)
    if( jsonify ):
        res = res.replace("'", '"')
    return res

def store_metadata(*, path, metadata):
    def lean(d):
        omit = ['url', 'filename', 'ext']
        u = {k:v for k,v in d.items() if k not in omit}
        u['source'] = os.path.join(d['url'], d['filename'])
        return u
    
    res = {}
    for k, v in metadata.items():
        res[k] = {}
        for k1, v1 in v.items():
            res[k][k1] = lean(v1)
        json_path = os.path.join(path, k, 'metadata.json')
        res_str = prettify_dict(res, jsonify=True) 
        sep = 80*'*' + '\n'
        s = sep 
        s += f'Storing metadata for {k} in {json_path}\n'
        s += res_str + f'\n{sep}\n'
        print(s)
        with open(json_path, 'w') as f:
            f.write(res_str)
    
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

def parse_path(path):
    if( path is None or path.startswith('conda') ):
        if( path == 'conda' ):
            path = 'conda/data'
        else:
            path = path.replace('conda', os.environ['CONDA_PREFIX'])
    elif( path.startswith('pwd') ):
        path = path.replace('pwd', os.getcwd())
    else:
        path = os.path.join(os.getcwd(), path)
    return path

def fetch_and_convert_data(
    *,
    subset='all',
    path=os.getcwd(),
    check=False
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    datasets = {
        'marmousi': {
            'url': 'https://www.geoazur.fr/WIND/pub/nfs/FWI-DATA/' + \
                'GEOMODELS/Marmousi',
            'ext': 'bin',
            'ny': 2301,
            'nx': 751,
            'dy': 4.0,
            'dx': 4.0,
            'dt': 0.004,
            'd_src': 20,
            'fst_src': 10,
            'src_depth': 2,
            'd_rec': 6,
            'fst_rec': 0,
            'rec_depth': 2,
            'd_intra_shot': 0,
            'freq': 25,
            'peak_time': 1.5 / 25,
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

def get_data2(*, field, path=None, allow_none=False):
    if( path is None or path.startswith('conda') ):
        if( path == 'conda' ):
            path = 'conda/data'
        else:
            path = path.replace('conda', os.environ['CONDA_PREFIX'])
    elif( path.startswith('pwd') ):
        path = path.replace('pwd', os.getcwd())
    else:
        path = os.path.join(os.getcwd(), path)

    field_file = os.path.join(path, f'{field}.pt')
    if( os.path.exists(path) ):
        try:
            return torch.load(field_file)
        except FileNotFoundError:
            if( allow_none ):
                print(f'File {field}.pt not found in {path}, return None')
                return None
            print(
                f'File {field}.pt not found in {path}' +
                f'\n    Delete {path} and try again'
            )
            raise
    subset = path.split('/')[-1]
    dummy_path = '/'.join(path.split('/')[:-1])

    if( os.path.exists(path) ):
        try:
            return torch.load(field_file)
        except FileNotFoundError:
            print(
                f'File {field}.pt not found in {path}' +
                f'\n    Delete {path} and try again'
            )
            raise
    fetch_and_convert_data(subset=subset, path=dummy_path)
    return torch.load(field_file)

def get_metadata(*, path):
    path = parse_path(path)
    return eval(open(f'{path}/metadata.json', 'r').read())

def get_primitives(d):
    prim_list = [int, float, str, bool]
    omit_keys = ['source', 'url', 'filename', 'ext']
    def helper(data, runner):
        for k,v in data.items():
            if( k in omit_keys ): continue
            if( type(v) == dict ):
                runner = helper(v, runner)
            elif( type(v) in prim_list ):
                if( k in runner.keys() and runner[k] != v ):
                    raise ValueError(f'Primitive type mismatch for {k}')
                else:
                    runner[k] = v
        return runner
    return helper(d, {})

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

class DataFactory(ABC):
    @auto_path(make_dir=False)
    def __init__(self, *, path):
        self.path = path

    def _manufacture_data(self, *, metadata, **kw):
        d = metadata

        if( os.path.exists(self.path) ):
            print(
                f'{self.path} already exists...ignoring.'
                'If you want to regenerate data, delete this folder ' 
                'or specify a different path.'
            )
            return
        os.makedirs(self.path, exist_ok=False)
        
        fields = { k:v for k,v in d.items() if type(v) == dict }
        assert( list(fields.keys()) == ['vp', 'rho'] )
        for k,v in fields.items():
            if( 'filename' not in v ):
                v['filename'] = k

        def field_url(x):
            url_path = os.path.join(d['url'], fields[x]['filename'])
            return url_path + '.' + d['ext']
        
        for k, v in fields.items():
            web_data_file = os.path.join(self.path, k) + f'.{d["ext"]}'
            final_data_file = os.path.join(self.path, k) + '.pt'
            cmd = f'curl {field_url(k)} --output {web_data_file}'
            header = f'ATTEMPT: {cmd}'
            stars = len(header) * '*'
            print(f'\n{stars}\nATTEMPT: {cmd}')
            os.system(cmd)
            print(f'SUCCESS\n{stars}\n')
            any_to_torch(
                input_path=web_data_file,
                output_path=final_data_file,
                **{**d, **v}
            )
            os.system(f'rm {web_data_file}')
            d[k] = torch.load(final_data_file)

        self.generate_derived_data(data=d, **kw)
        return d
    
    @abstractmethod
    def generate_derived_data(self, *, data, **kw):
        pass
