<<<<<<< HEAD
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
=======
import os
import argparse
import torch
from obspy.io.segy.segy import _read_segy

def segy_to_tensor(file_path, device, transpose):
    # Read the SEGY file using obspy
    segy_data = _read_segy(file_path, headonly=True)
    
    # Convert traces to tensors
    traces = [torch.tensor(trace.data, device=device) for trace in segy_data.traces]
    
    stacked_tensor = torch.stack(traces)
    
    return stacked_tensor if not transpose else torch.transpose(stacked_tensor, 0, 1)

def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description="Convert SEGY files to PyTorch tensors")
    parser.add_argument("--folder", type=str, required=True, help="Folder to search for all SEGY files.")
    parser.add_argument("--suffix", type=str, default="segy", help="Extension to search for (e.g., segy or sgy).")
    parser.add_argument("--device", type=str, default="cpu", help="Device to store the tensor (e.g., cpu, cuda:0).")
    parser.add_argument('--transpose', action='store_true', help='Transpose the data?')
    
    args = parser.parse_args()
    
    # Validate folder path
    if not os.path.exists(args.folder):
        raise ValueError(f"Folder {args.folder} does not exist.")
    
    # Define the output folder
    output_folder = os.path.join(args.folder, "torch_conversions")
    os.makedirs(output_folder, exist_ok=True)
    
    # Define device
    device = torch.device(args.device)
    
    # Iterate through files in folder with the given suffix
    for file in os.listdir(args.folder):
        if file.endswith(args.suffix):
            file_path = os.path.join(args.folder, file)
            tensor = segy_to_tensor(file_path, device, args.transpose)
            
            # Save the tensor
            output_tensor_path = os.path.join(output_folder, f"{os.path.splitext(file)[0]}.pt")
            torch.save(tensor, output_tensor_path)
            print(f"Saved tensor for {file} to {output_tensor_path}")

if __name__ == "__main__":
>>>>>>> 93b54993ff268c8a45c66909ca685592fb97a043
    main()

