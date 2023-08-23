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
        base = {k:v for k,v in folder_meta.items() if type(v) == str}
        files = {k:v for k,v in folder_meta.items() if type(v) == dict}
        for filename, file_meta in files.items():
            d[folder][filename] = {**base, **file_meta}
            if 'filename' not in d[folder][filename].keys():
                d[folder][filename]['filename'] = \
                    filename + d[folder][filename]['ext']
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
    nx
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
    for folder, info in d.items():
        # Create directory if it doesn't exist
        os.makedirs(folder, exist_ok=True)
        
        for file, meta in info.items():
            url = meta['url'] + meta['filename']

            response = requests.get(url, stream=True)
            
            if response.status_code == 200:
                file_path = os.path.join(folder, meta['filename']) 
                with open(file_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
            if( unzip and meta['filename'].endswith('.gz') ):
                os.system(f'gunzip {file_path}')
            else:
                print(f"Failed to fetch data from {url}. Status code: {response.status_code}")
            
def convert_data(d, *, device='cpu'):
    for folder, info in d.items():
        for filename in info.keys():
            if( filename == 'meta' ): continue
            file_path = os.path.join(folder, filename)
            torch_array = segy_to_torch(
                file_path, 
                device=device, 
                transpose=True,
                **info['meta'].get(filename, {})
            )
            torch_val = filename.replace('.segy', '.pt').replace('.sgy', '.pt')
            torch_folder = os.path.join(folder, 'torch_conversions')
            os.makedirs(torch_folder, exist_ok=True)
            torch_val = os.path.join(torch_folder, torch_val)
            torch.save(torch_array, torch_val)
            print(f'done! Saved to {torch_val}')
def main():
    parser = argparse.ArgumentParser(description='Download and convert data')

    datasets = {
        'marmousi': {
            'url': 'https://www.geoazur.fr/WIND/pub/nfs/FWI-DATA/GEOMODELS/',
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

    
        
    # # Create an argument parser
    # parser = argparse.ArgumentParser(description='Convert a SEGY file to a numpy array.')

    # # Add an argument for the file path
    # parser.add_argument('--folder', help='The path of the SEGY file to convert.')
    # parser.add_argument('--suffix', type=str, default='segy')
    # parser.add_argument('--device', type=str, default='cpu')
    # parser.add_argument('--transpose', action='store_true')

    # # Parse the arguments
    # args = parser.parse_args()

    # files = sco('find %s -name "*.%s"'%(args.folder, args.suffix))
    # device = torch.device(args.device)
    # print('Found files\n%s'%('\n'.join(files)))
    # for e in files:
    #     print(f'Converting {e}...', end='', file=sys.stderr)
    #     # Call the function with the file path from the arguments
    #     torch_array = segy_to_torch(e, device=device, transpose=args.transpose)
    #     torch_val = e.split('/')[-1].replace('.%s'%args.suffix, '.pt')
    #     torch_val = os.path.join(args.folder, 'torch_conversions', torch_val)
    #     torch.save(torch_array, torch_val)
    #     print(f'done! Saved to {torch_val}')

if __name__ == '__main__':
    main()

