import argparse
import numpy as np
from obspy.io.segy.segy import _read_segy
import torch
from subprocess import check_output as co
import os
import time
import sys
import requests
from subprocess import CalledProcessError
import torch
from ..swiffer import sco
import re
import copy 

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


