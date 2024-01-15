import torch
from torchcubicspline import natural_cubic_spline_coeffs, NaturalCubicSpline
import matplotlib.pyplot as plt
import numpy as np
from itertools import product


def unbatch_splines(coeffs):
    assert len(coeffs) == 5
    assert coeffs[1].shape[-1] == 1
    target_shape = coeffs[1].shape[:-2]
    res = np.empty(target_shape, dtype=object)
    for idx in product(*map(range, target_shape)):
        curr = (
            coeffs[0],
            coeffs[1][idx],
            coeffs[2][idx],
            coeffs[3][idx],
            coeffs[4][idx],
        )
        res[idx] = NaturalCubicSpline(curr)
    return res


def unbatch_spline_eval(splines, t):
    assert t.shape[:-1] == splines.shape
    t = t.unsqueeze(-1)
    res = torch.empty(t.shape)
    for idx in product(*map(range, t.shape[:-2])):
        # t_idx = (*idx, slice(None), slice(0, 1))
        res[idx] = splines[idx].evaluate(t[idx]).squeeze(-1)
    return res


nt = 100
n_batches = 25
dims = 1

a, b = 0.0, 1.0
t = torch.linspace(a, b, nt)
# (2, 1) are batch dimensions. 7 is the time dimension
# (of the same length as t). 3 is the channel dimension.

freq_min = 0.0, 3.0
freqs = (
    torch.linspace(freq_min[0], freq_min[1], n_batches)
    .unsqueeze(0)
    .unsqueeze(0)
)
t_expanded = t.unsqueeze(0).unsqueeze(-1)
x = torch.sin(2 * torch.pi * freqs * t_expanded).permute(2, 1, 0)

coeffs = natural_cubic_spline_coeffs(t, x)
# splines = NaturalCubicSpline(coeffs)
splines = unbatch_splines(coeffs)
t_stacked = torch.stack([t] * n_batches)
# t_stacked = t_stacked + 0.1 * torch.rand_like(t_stacked)
# res = splines.evaluate(t_stacked)
t_stacked = t_stacked + 0.1 * torch.rand_like(t_stacked)
t_stacked = t_stacked.sort(dim=-1).values
res = unbatch_spline_eval(splines, t_stacked)


x = x.squeeze()
res = res.squeeze()

# input(x.shape)
# input(t.shape)
# input(res.shape)

num_samples = 200
num_samples = min(num_samples, n_batches)

for i in range(num_samples):
    print(f'Plotting {i}')
    plt.plot(t, x[i], linestyle='-', label='raw', color='blue')
    plt.plot(t_stacked[i], res[i], linestyle='-.', label='spline', color='red')
    plt.ylim(-1.0, 1.0)
    plt.legend()

    plt.savefig(f'test{i}.jpg')
    plt.clf()

import os

os.system('convert -delay 20 -loop 0 $(ls -t *.jpg) test.gif')
os.system('rm *.jpg')
