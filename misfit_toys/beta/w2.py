import os
from time import time

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from mh.core import torch_stats
from torchcubicspline import NaturalCubicSpline as NCS
from torchcubicspline import natural_cubic_spline_coeffs as ncs

# Set print options or any global settings
torch.set_printoptions(precision=10, callback=torch_stats())


def simple_coeffs(t, x):
    coeffs = ncs(t, x)
    right = F.pad(torch.stack([e.squeeze() for e in coeffs[1:]], dim=0), (1, 0))
    return torch.cat([coeffs[0][None, :], right], dim=0)


# def compute_spline_coeffs(i, shared_data, shared_results, t):
#     if i % 1 == 0:
#         print(f'{i} / {len(shared_data)}', flush=True)
#     shared_results[i] = simple_coeffs(shared_data[i], t)


def compute_spline_coeffs(start, end, shared_data, shared_results, t):
    # out_file = open(f"worker_{rank}.txt", "w")
    # out_file.write(f'Total iters: {end-start}\n\n')
    for i in range(start, end):
        # if verbose and (i - start) % 100 == 0:
        #     out_file.write(f"{i - start}\n")
        #     out_file.flush()
        shared_results[i] = simple_coeffs(shared_data[i], t).T


def parallel_for(*, obs_data, t, workers=None):
    if workers is None:
        workers = os.cpu_count() - 1
    shared_data = obs_data.share_memory_()
    shared_results = torch.empty(*obs_data.shape, 5).share_memory_()
    processes = []
    delta = len(obs_data) // workers
    for i in range(workers):
        start = i * delta
        end = min((i + 1) * delta, len(obs_data))
        p = mp.Process(
            target=compute_spline_coeffs,
            args=(start, end, shared_data, shared_results, t),
        )
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    return shared_results


def softplus_renorm(u, t):
    softp = torch.nn.Softplus(beta=1, threshold=20)
    cdf = torch.cumulative_trapezoid(softp(u), t.squeeze(), dim=-1)
    cdf = F.pad(cdf, (1, 0))
    cdf = cdf / cdf[:, -1].unsqueeze(-1)
    return cdf


def quantile_spline_coeffs(*, input_path, output_path, renorm, workers=None):
    mp.set_start_method('spawn')  # Necessary for PyTorch multiprocessing
    if os.path.exists(output_path):
        return torch.load(output_path)

    obs_data = torch.load(input_path)
    obs_data = obs_data.reshape(-1, obs_data.shape[-1])
    t = torch.linspace(0, 1.1, obs_data.shape[-1]).unsqueeze(-1)
    robs = renorm(obs_data, t)

    # Process in parallel using shared memory
    results = parallel_for(obs_data=robs, t=t, workers=workers)
    results = results.permute(0, 2, 1)  # Reshape if necessary

    torch.save(results, output_path)
    return results


def quantile_splines(coeffs):
    splines = np.empty(coeffs.shape[0], dtype=object)
    for i in range(coeffs.shape[0]):
        c = [coeffs[i, 0, :]]
        c.extend(
            [coeffs[i, j, 1:].unsqueeze(-1) for j in range(1, coeffs.shape[1])]
        )
        splines[i] = NCS(c)
    return splines


def fetch_splines(*, input_path, output_path, renorm, workers=None):
    coeffs = quantile_spline_coeffs(
        input_path=input_path,
        output_path=output_path,
        renorm=renorm,
        workers=workers,
    )
    return quantile_splines(coeffs)


def main():
    input_path = f"{os.environ['CONDA_PREFIX']}/data/marmousi/obs_data.pt"
    output_path = "out.pt"

    start_time = time()
    fetch_splines(
        input_path=input_path,
        output_path=output_path,
        renorm=softplus_renorm,
        workers=os.cpu_count() - 1,
    )
    print(f'Processing time: {time() - start_time}s')


if __name__ == "__main__":
    # mp.set_start_method('spawn')  # Necessary for PyTorch multiprocessing
    main()
