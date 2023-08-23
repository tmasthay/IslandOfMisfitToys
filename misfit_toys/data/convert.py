import argparse
import numpy as np
from obspy.io.segy.segy import _read_segy
import torch
from subprocess import check_output as co
import os
import time
import sys

def sco(s, split=True):
    u = co(s, shell=True).decode('utf-8')
    if( split ): return u.split('\n')[:-1]
    else: return u

def segy_to_torch(file_path, *, device, transpose=False, out=sys.stderr):

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

def main():
    # Create an argument parser
    parser = argparse.ArgumentParser(description='Convert a SEGY file to a numpy array.')

    # Add an argument for the file path
    parser.add_argument('--folder', help='The path of the SEGY file to convert.')
    parser.add_argument('--suffix', type=str, default='segy')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--transpose', action='store_true')

    # Parse the arguments
    args = parser.parse_args()

    files = sco('find %s -name "*.%s"'%(args.folder, args.suffix))
    device = torch.device(args.device)
    print('Found files\n%s'%('\n'.join(files)))
    for e in files:
        print(f'Converting {e}...', end='', file=sys.stderr)
        # Call the function with the file path from the arguments
        torch_array = segy_to_torch(e, device=device, transpose=args.transpose)
        torch_val = e.split('/')[-1].replace('.%s'%args.suffix, '.pt')
        torch_val = os.path.join(args.folder, 'torch_conversions', torch_val)
        torch.save(torch_array, torch_val)
        print(f'done! Saved to {torch_val}')

if __name__ == '__main__':
    main()

