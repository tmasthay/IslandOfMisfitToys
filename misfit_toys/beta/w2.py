import os
import torch
from concurrent.futures import ProcessPoolExecutor
from torchcubicspline import natural_cubic_spline_coeffs as ncs
import torch.nn.functional as F
from time import time
from mh.core import torch_stats

# Set print options or any global settings
torch.set_printoptions(callback=torch_stats())


def simple_coeffs(t, x):
    coeffs = ncs(t, x)
    right = F.pad(torch.stack([e.squeeze() for e in coeffs[1:]], dim=0), (1, 0))
    return torch.cat([coeffs[0][None, :], right], dim=0)


def compute_spline_coeffs(i, *, obs_data, t):
    if i % 100 == 0:
        print(f'{i} / {obs_data.shape[0]}')
    return simple_coeffs(obs_data[i], t)


def parallel_for(func, iter_data, args=(), kwargs={}, workers=None):
    if workers is None:
        workers = os.cpu_count()
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(func, i, *args, **kwargs) for i in iter_data]
        results = [future.result() for future in futures]
    return results


def quantile_spline_coeffs(input_path, output_path, chunk_size=None, workers=None):
    if os.path.exists(output_path):
        return torch.load(output_path)

    obs_data = torch.load(input_path)
    obs_data = obs_data.reshape(-1, obs_data.shape[-1])
    t = torch.linspace(0, 1.1, obs_data.shape[-1]).unsqueeze(-1)
    
    # Determine the total number of data points
    total_data_points = obs_data.shape[0]
    results = []

    # Determine chunk size
    if chunk_size is None:
        chunk_size = total_data_points  # Process all data at once

    # Process in chunks
    for start in range(0, total_data_points, chunk_size):
        end = min(start + chunk_size, total_data_points)
        current_indices = range(start, end)
        current_results = parallel_for(
            compute_spline_coeffs,
            current_indices,
            kwargs={"obs_data": obs_data, "t": t},
            workers=workers
        )
        results.extend(current_results)
    
    # Combine the results
    v = torch.stack(results)
    v = v.permute(0, 2, 1)  # Reshape if necessary

    torch.save(v, output_path)
    return v


def main():
    input_path = "/home/tyler/miniconda3/envs/dw/data/marmousi/obs_data.pt"  # Modify as needed
    output_path = "out.pt"  # Modify as needed

    start_time = time()
    v = quantile_spline_coeffs(input_path, output_path, workers=None, chunk_size=None)
    print(f"Processing time: {time() - start_time}s")
    print(v)


if __name__ == "__main__":
    main()
