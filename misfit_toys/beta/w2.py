import os
from time import time

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from matplotlib import pyplot as plt
from mh.core import DotDict, torch_stats
from mh.typlotlib import get_frames_bool, save_frames
from returns.curry import curry
from torchcubicspline import NaturalCubicSpline as NCS
from torchcubicspline import natural_cubic_spline_coeffs as ncs

from misfit_toys.utils import bool_slice, clean_idx, get_pydict

# Set print options or any global settings
torch.set_printoptions(precision=10, callback=torch_stats('all'))


def get_cpus(*, spare=0):
    return os.cpu_count() - spare


def simple_coeffs(t, x):
    coeffs = ncs(t, x)
    right = F.pad(
        torch.stack([e.squeeze() for e in coeffs[1:]], dim=0), (1, 0), value=100
    )
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
    delta = int(-1.0 * (-len(obs_data) // workers))
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


def softplus(u, t):
    softp = torch.nn.Softplus(beta=1.0, threshold=20)
    return softp(u)


def prob(*, data, t):
    cdf = torch.cumulative_trapezoid(data, t.squeeze(), dim=-1)
    cdf = F.pad(cdf, (1, 0))
    integration_constants = cdf[:, -1].unsqueeze(-1)
    cdf = cdf / integration_constants
    pdf = data / integration_constants
    return pdf, cdf


def quantile_spline_coeffs(*, input_path, output_path, transform, workers=None):
    meta = get_pydict(input_path, as_class=True)
    t = torch.linspace(0, (meta.nt - 1) * meta.dt, meta.nt).unsqueeze(-1)
    # t = torch.linspace(0, 1.0, meta.nt).unsqueeze(-1)
    if os.path.exists(output_path):
        return torch.load(output_path), t
    else:
        os.makedirs(os.path.dirname(output_path))

    mp.set_start_method('spawn')  # Necessary for PyTorch multiprocessing
    data_path = os.path.join(input_path, 'obs_data.pt')
    obs_data = torch.load(data_path)
    obs_data = obs_data.reshape(-1, obs_data.shape[-1])

    robs = transform(obs_data, t)
    _, cdf = prob(data=robs, t=t)

    # Process in parallel using shared memory
    results = parallel_for(obs_data=cdf, t=t, workers=workers)
    results = results.permute(0, 2, 1)  # Reshape if necessary

    torch.save(results, output_path)
    return results, t


def quantile_splines(coeffs):
    splines = np.empty(coeffs.shape[0], dtype=object)
    for i in range(coeffs.shape[0]):
        c = [coeffs[i, 0, :]]
        c.extend(
            [coeffs[i, j, 1:].unsqueeze(-1) for j in range(1, coeffs.shape[1])]
        )
        splines[i] = NCS(c)
    return splines


def fetch_quantile_splines(*, input_path, output_path, transform, workers=None):
    coeffs, t = quantile_spline_coeffs(
        input_path=input_path,
        output_path=output_path,
        transform=transform,
        workers=workers,
    )
    return quantile_splines(coeffs), t


class Wasserstein(torch.nn.Module):
    def __init__(self, *, input_path, output_path, transform, workers=None):
        super().__init__()
        self.transform = transform
        self.input_path = input_path
        self.output_path = output_path
        self.coeffs, self.t = quantile_spline_coeffs(
            input_path=input_path,
            output_path=output_path,
            transform=transform,
            workers=workers,
        )
        # self._modules = {}

    def to(self, device):
        self.t = self.t.to(device)
        self.coeffs = self.coeffs.to(device)
        self.splines = quantile_splines(self.coeffs)
        return self

    def forward(self, f):
        f = f.reshape(-1, f.shape[-1])
        fpos = self.transform(f, self.t.squeeze())
        pdf, cdf = prob(data=fpos, t=self.t.squeeze())
        runner = torch.tensor(0.0, requires_grad=True)
        for i in range(cdf.shape[0]):
            cut_cdf_left = torch.argmax((cdf[i] > 0.3).float())
            cut_cdf_right = torch.argmin((cdf[i] < 0.7).float())
            cut_cdf = cdf[i, cut_cdf_left:cut_cdf_right]
            cut_t = self.t[cut_cdf_left:cut_cdf_right]
            cut_pdf = pdf[i, cut_cdf_left:cut_cdf_right]
            T = self.splines[i].evaluate(cut_cdf).squeeze()
            integrand = (
                self.t[cut_cdf_left:cut_cdf_right].squeeze() - T
            ) ** 2 * cut_pdf
            res = torch.trapz(integrand, cut_t.squeeze())
            runner = runner + res
        return runner


def plotter(*, data, idx, fig, axes):
    plt.clf()
    plt.subplot(*data.shape, 1)
    plt.plot(data.t, data.obs_data[idx])
    plt.title('Raw Data')

    plt.subplot(*data.shape, 2)
    plt.plot(data.t, data.pdf[idx])
    plt.title('PDF')

    plt.subplot(*data.shape, 3)
    plt.plot(data.t, data.cdf[idx] - data.lin_cdf)
    plt.title('CDF')

    plt.subplot(*data.shape, 4)
    plt.plot(
        data.p, data.splines[idx[0]].evaluate(data.p).squeeze() - data.lin_q
    )
    plt.title('Quantile')

    # plt.subplot(*data.shape, 5)
    # plt.plot(data.t, data.res[idx].squeeze() - data.t.squeeze())
    # plt.title(r'$Q(F(t)) \approx t$')

    plt.subplot(*data.shape, 5)
    plt.plot(data.t, data.integrand[idx])
    plt.title('Integrand')

    plt.subplot(*data.shape, 6)
    plt.plot(range(data.w2_dist.shape[0]), data.w2_dist)
    plt.plot([idx[0]], [data.w2_dist[idx[0]]], 'ro')
    plt.title('Loss')

    plt.suptitle(f'{clean_idx(idx)}')
    plt.tight_layout()


def plot_test(*, input_path, output_path, transform, workers=None):
    splines, t = fetch_quantile_splines(
        input_path=input_path,
        output_path=output_path,
        workers=workers,
        transform=transform,
    )
    eta = 0.0
    obs_data = torch.load(f'{input_path}/obs_data.pt')
    obs_data = obs_data.reshape(-1, obs_data.shape[-1])
    obs_data = obs_data + torch.rand_like(obs_data) * eta

    robs = transform(obs_data, t)
    pdf, cdf = prob(data=robs, t=t)
    # pdf = robs / torch.trapz(robs)

    lin_cdf = (1.0 / t[-1] * t).squeeze()

    p = torch.linspace(0.0, 1.0, 1000)
    lin_q = (t[-1] * p).squeeze()
    obs_data = obs_data[: splines.shape[0]]
    res = torch.rand_like(obs_data)
    for (i, s), e in zip(enumerate(splines), cdf):
        res[i] = s.evaluate(e).squeeze()

    integrand = (res - t.reshape(1, -1)) ** 2 * pdf
    w2_dist = torch.trapz(integrand, t.squeeze(), dim=-1)

    shape = (3, 2)
    fig, axes = plt.subplots(*shape, figsize=(10, 10))
    iter = bool_slice(*res.shape, strides=[100, 1], none_dims=[-1])
    # input(DotDict(locals()))
    frames = get_frames_bool(
        data=DotDict(locals()), iter=iter, fig=fig, axes=axes, plotter=plotter
    )
    save_frames(frames, path='res.gif')
    print('res.gif')


def main():
    # input_path = (
    #     f"{os.environ['CONDA_PREFIX']}/data/marmousi/deepwave_example/shots16"
    # )
    data_path = 'data/marmousi/deepwave_example/shots16'
    os.makedirs(data_path, exist_ok=True)
    input_path = os.path.join(os.environ['CONDA_PREFIX'], data_path)
    output_path = os.path.join(data_path, 'splines.pt')

    plot_test(
        input_path=input_path,
        output_path=output_path,
        workers=os.cpu_count() - 1,
        transform=softplus,
    )


if __name__ == "__main__":
    # mp.set_start_method('spawn')  # Necessary for PyTorch multiprocessing
    main()
