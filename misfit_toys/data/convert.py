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

def sco(s, split=True):
    try:
        u = co(s, shell=True).decode('utf-8')
        if( split ): return u.split('\n')[:-1]
        else: return u
    except CalledProcessError:
        return None

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
    file_path, 
    *, 
    device='cpu', 
    transpose=False, 
    out=sys.stdout,
    **kw
):
    print('READING in SEGY file "%s"'%file_path, file=out)

    # Read the SEGY file
    stream = _read_segy(file_path)

    print('DONE reading SEGY file "%s"'%file_path, file=out)

    # Create a list to hold the data
    data_list = []

    N = len(stream.traces)

    t = time.time()
    # Loop through each trace in the stream
    for (i,trace) in enumerate(stream.traces):
        if( i == 0 ): print('READING first trace', file=out)
        # Add the trace's data to the list
        data_list.append(trace.data)

        elapsed = time.time() - t
        avg = elapsed / (i+1)
        rem = (N-(i+1)) * avg
        print(
            'Elapsed (s): %f, Remaining estimate (s): %f'%(elapsed, rem), 
            file=out
        )

    print('Converting list to pytorch tensor (last step)')
    # Convert the list to a numpy array and return it
    return torch.Tensor(data_list).to(device) if not transpose else \
        torch.Tensor(data_list).transpose(0,1).to(device)

def bin_to_torch(
    file_path, 
    *, 
    device='cpu', 
    transpose=False, 
    out=sys.stdout,
    ny,
    nx,
    **kw
):
    u = torch.from_file(file_path, size=ny*nx)
    torch.save(u.reshape(ny,nx).to(device), file_path.replace('.bin', '.pt'))

def any_to_torch(
    file_path,
    *,
    device='cpu',
    transpose=False,
    out=sys.stdout,
    **kw
):
    if( file_path.endswith('.segy') or file_path.endswith('.sgy') ):
        segy_to_torch(
            file_path, 
            device=device, 
            transpose=transpose, 
            out=out,
            **kw
        )
    elif( file_path.endswith('.bin') ):
        bin_to_torch(
            file_path,
            device=device,
            transpose=transpose,
            out=out,
            **kw
        )
    else:
        raise ValueError(f'Unknown file type: {file_path}')
     
def fetch_data(d, *, unzip=True):
    convert_search = dict()
    for folder, info in d.items():
        convert_search[folder] = []
        # Create directory if it doesn't exist
        os.makedirs(folder, exist_ok=True)
        
        for file, meta in info.items():
            url = meta['url'] + meta['filename']
            file_path = f'{folder}/{meta["filename"]}'
            print(f'ATTEMPT: {url} -> {file_path}') 
            os.system(f'curl {url} --output {file_path}') 
            if( unzip and meta['filename'].endswith('.gz') ):
                os.system(f'gunzip {file_path}')
                d[folder][file]['filename'] = \
                    d[folder][file]['filename'].replace('.gz', '')
    return d

def convert_data(d, *, device='cpu'):
    for folder, files in d.items():
        for field, meta in files.items():
            any_to_torch(
                file_path=(
                    os.path.join(folder, meta['filename'])
                ),
                **meta
            )
            os.system(f'mv {os.path.join(folder, meta["filename"])} ' + 
                f'{field}.pt'
            )
        os.system('rm %s/*.%s'%(
                folder,
                f' {folder}/*.'.join(['bin', 'sgy', 'segy', 'gz'])
            )
        )

def main():
    parser = argparse.ArgumentParser(description='Download and convert data')

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
            'url': 'http://www.agl.uh.edu/downloads/downloads/',
            'ext': 'segy',
            'vp': {'filename': 'vp_marmousi-ii.segy.gz'},
            'vs': {'filename': 'vs_marmousi-ii.segy.gz'},
            'rho': {'filename': 'density_marmousi-ii.segy.gz'}
        },
        'DAS': {
            'url': 'https://ddfe.curtin.edu.au/7h0e-d392/',
            'ext': 'sgy',
            'das_curtin': {'filename': '2020_GeoLab_WVSP_DAS_wgm.sgy'},
            'geophone_curtin.sgy': {
                'filename': '2020_GeoLab_WVSP_geophone_wgm.sgy'
            },
        }
    }
    datasets = expand_metadata(datasets)
    parser.add_argument(
        '--datasets',
        type=str, 
        nargs='+', 
        choices=(list(datasets.keys()) + ['all']),
        default=list(datasets.keys()),
        help='Dataset choices: [%s, all]'%(', '.join(datasets.keys()))
    )
    args = parser.parse_args()
   
    if( 'all' not in args.datasets 
       and set(args.datasets) != set(datasets.keys()) 
    ):
        datasets = {k:v for k,v in datasets.items() if k in args.datasets} 

    fetch_data(datasets)
    convert_data(datasets)

if __name__ == '__main__':
    main()

