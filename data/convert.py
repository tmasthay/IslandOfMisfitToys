import os
import argparse
import torch
from obspy.io.segy.segy import _read_segy

def segy_to_tensor(file_path, device):
    # Read the SEGY file using obspy
    segy_data = _read_segy(file_path, headonly=True)
    
    # Convert traces to tensors
    traces = [torch.tensor(trace.data, device=device) for trace in segy_data.traces]
    
    # Stack the individual traces to form a single tensor
    stacked_tensor = torch.stack(traces)
    
    return stacked_tensor

def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description="Convert SEGY files to PyTorch tensors")
    parser.add_argument("--folder", type=str, required=True, help="Folder to search for all SEGY files.")
    parser.add_argument("--suffix", type=str, default="segy", help="Extension to search for (e.g., segy or sgy).")
    parser.add_argument("--device", type=str, default="cpu", help="Device to store the tensor (e.g., cpu, cuda:0).")
    
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
            tensor = segy_to_tensor(file_path, device)
            
            # Save the tensor
            output_tensor_path = os.path.join(output_folder, f"{os.path.splitext(file)[0]}.pt")
            torch.save(tensor, output_tensor_path)
            print(f"Saved tensor for {file} to {output_tensor_path}")

if __name__ == "__main__":
    main()

