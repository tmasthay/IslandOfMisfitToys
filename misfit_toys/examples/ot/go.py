from itertools import product
from time import time
from typing import Callable, List

import matplotlib.pyplot as plt
import numpy as np
import torch

# from pympler import asizeof
from torchcubicspline import NaturalCubicSpline, natural_cubic_spline_coeffs


def cum_trap(*args, preserve_dims=True, **kw):
    u = torch.cumulative_trapezoid(*args, **kw)
    if preserve_dims:
        dim = kw.get('dim', -1)
        if dim < 0:
            dim = len(u.shape) + dim
        v = torch.zeros(
            [e if i != dim else e + 1 for i, e in enumerate(u.shape)]
        )
        slices = [
            slice(None) if i != dim else slice(1, None)
            for i in range(len(u.shape))
        ]
        v[slices] = u
        return v
    return u


# Function to compute true_quantile for an arbitrary shape torch tensor along its last dimension
def true_quantile(pdf, x, p, *, dx=None):
    if len(pdf.shape) == 1:
        if dx is not None:
            cdf = cum_trap(pdf, dx=dx, dim=-1)
        else:
            cdf = cum_trap(pdf, x, dim=-1)
        indices = torch.searchsorted(cdf, p)
        indices = torch.clamp(indices, 0, len(x) - 1)
        return x[indices], cdf
    else:
        # Initialize an empty tensor to store the results
        result_shape = pdf.shape[:-1]
        results = torch.empty(
            result_shape + (2, pdf.shape[-1]), dtype=torch.float32
        )

        # Loop through the dimensions
        for idx in product(*map(range, result_shape)):
            pdf_slice = pdf[idx]
            x_slice, cdf_slice = true_quantile(pdf_slice, x, p, dx=dx)
            results[idx] = torch.stack([x_slice, cdf_slice], dim=0)

        num_dims = len(results.shape)
        permutation = (
            [num_dims - 2] + list(range(num_dims - 2)) + [num_dims - 1]
        )

        return results.permute(*permutation)


def rand_elem(shape):
    tensors = [
        torch.randint(0, e, size=(1,)).item() if e else None for e in shape
    ]
    slices = [slice(None) if i is None else slice(i, i + 1) for i in tensors]
    return slices


def renorm_quantiles(
    obs: torch.Tensor,
    renorm_func: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    p: torch.Tensor,
):
    return true_quantile(renorm_func(obs), x, p=p)


def w2_integrand(*, q, pdf, x, dx=None):
    if dx is None:
        cdf = cum_trap(pdf, x, dim=-1)
    else:
        cdf = cum_trap(pdf, dx=dx, dim=-1)
    off_diag = torch.transpose(q.evaluate(cdf), -2, -1).squeeze(-2)
    integrand = (x - off_diag) ** 2 * pdf
    return integrand


def get_renorm(t, s):
    def abs_renorm(f):
        u = torch.abs(f)
        return u / torch.trapz(u, t, dim=-1).unsqueeze(-1)

    def square_renorm(f):
        u = f**2
        return u / torch.trapz(u, t, dim=-1).unsqueeze(-1)

    def split_renorm(f):
        f_pos, f_neg = torch.clamp(f, min=0.0), -torch.clamp(f, max=0.0)
        f_pos_renorm = f_pos / torch.trapz(f_pos, t, dim=-1).unsqueeze(-1)
        f_neg_renorm = f_neg / torch.trapz(f_neg, t, dim=-1).unsqueeze(-1)
        return torch.stack([f_pos_renorm, f_neg_renorm], dim=0)

    d = {'abs': abs_renorm, 'square': square_renorm, 'split': split_renorm}
    return d[s]


def main():
    training_data = torch.load("out/out_record.pt")
    # training_data = training_data[0:1, 0:2].squeeze()
    shifted_data = torch.roll(training_data, shifts=10, dims=-1)

    a, b = 0.0, 2.0
    t = torch.linspace(a, b, training_data.shape[-1])
    p = torch.linspace(0, 1.0, training_data.shape[-1])
    renorm_func = get_renorm(t, 'abs')

    preprocess_time = time()
    training_data_renorm = renorm_func(training_data)
    shifted_data_renorm = renorm_func(shifted_data)
    cdf_data = cum_trap(training_data_renorm, dim=-1, dx=t[1] - t[0])
    cdf_shifted_data = cum_trap(shifted_data_renorm, dim=-1, dx=t[1] - t[0])
    quantiles = renorm_quantiles(training_data, renorm_func, t, p)
    preprocess_time = time() - preprocess_time

    offline_quantile = time()
    # squeezed_quantile                   s = quantiles[0].reshape(
    #     torch.prod(torch.tensor(quantiles[0].shape[:-1])), -1, 1
    # )
    squeezed_quantiles = quantiles[0].unsqueeze(-1)
    input(f'squeezed_quantiles.shape: {squeezed_quantiles.shape}')
    coeffs = natural_cubic_spline_coeffs(t, squeezed_quantiles)
    input([e.shape for e in coeffs])
    quantile_splines = NaturalCubicSpline(
        natural_cubic_spline_coeffs(t, squeezed_quantiles)
    )
    input(quantile_splines[0])
    input(quantile_splines.evaluate(torch.tensor(p[::3])).shape)
    offline_quantile = time() - offline_quantile

    online_quantile = time()
    # integrands = w2_integrand(
    #     q=quantile_splines, pdf=shifted_data, x=t, dx=t[1] - t[0]
    # )
    online_quantile = time() - online_quantile

    num_samples = 10
    plot_time = time()
    for sample_no in range(num_samples):
        shape = list(training_data.shape[:-1]) + [None]
        idx = rand_elem(shape)
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 3, 1)
        plt.plot(t, training_data[idx].reshape(-1), label='Raw Data')
        plt.plot(t, training_data_renorm[idx].reshape(-1), label='PDF')
        plt.legend()
        plt.title("PDF")

        plt.subplot(1, 3, 2)
        plt.plot(t, cdf_data[idx].reshape(-1), label='True CDF')
        plt.plot(t, cdf_shifted_data[idx].reshape(-1), label='Shifted CDF')
        plt.legend()
        plt.title("CDF")

        plt.subplot(1, 3, 3)
        plt.plot(p, quantiles[0][idx].reshape(-1), label='True Quantile')
        plt.title('Quantile')

        plt.savefig(f'quantile{sample_no}.jpg')
        print(f'Finished {sample_no} plots')
        plt.close()
    plot_time = time() - plot_time

    print(f'Preprocess time: {preprocess_time}')
    print(f'Offline quantile time: {offline_quantile}')
    print(f'Online quantile time: {online_quantile}')
    print(f'Plot time: {plot_time}')


# Example usage
def main2():
    # Load your training_data from a file using torch.load
    training_data = torch.load("out/out_record.pt")
    training_data = training_data[0:1, 0:2]

    noise_level = 0.0
    noise_level *= torch.max(torch.abs(training_data))
    # variance = noise_level * torch.rand_like(training_data)

    training_data += noise_level * torch.randn_like(training_data)

    a, b = 0.0, 1.0
    t = torch.linspace(a, b, training_data.shape[-1])
    p = torch.linspace(0, 1.0, training_data.shape[-1])
    # Assuming renorm_func has already been applied to training_data
    # training_data = (
    #     torch.abs(training_data)
    #     / torch.sum(torch.abs(training_data), dim=-1, keepdim=True)
    #     / (t[1] - t[0])
    # )
    dt = t[1] - t[0]

    def abs_renorm(f):
        u = torch.abs(f)
        return u / torch.sum(u, dim=-1, keepdim=True) / dt

    def square_renorm(f):
        u = f**2
        return u / torch.sum(u, dim=-1, keepdim=True) / dt

    def split_renorm(f):
        f_pos, f_neg = torch.clamp(f, min=0.0), -torch.clamp(f, max=0.0)
        f_pos_renorm = f_pos / torch.sum(f_pos, dim=-1, keepdim=True) / dt
        f_neg_renorm = f_neg / torch.sum(f_neg, dim=-1, keepdim=True) / dt
        return torch.stack([f_pos_renorm, f_neg_renorm], dim=0)

    chosen_renorm = abs_renorm
    # training_data_processing = renorm_quantiles(
    #     training_data, chosen_renorm, t, p
    # )
    renorm_data = chosen_renorm(training_data)
    training_data_processing = true_quantile(renorm_data, t, p)
    quantiles = training_data_processing[0]
    cdf = training_data_processing[1]

    if chosen_renorm == split_renorm:
        quantiles = quantiles.squeeze(1)
        cdf = cdf.squeeze(1)

    # quantile_perm = list(range(len(quantiles.shape)))
    # tmp = quantile_perm[-2]
    # quantile_perm[-2] = quantile_perm[-1]
    # quantile_perm[-1] = tmp
    # # quantiles = quantiles.permute(*quantile_perm)

    # Perform cubic spline approximation for each element of training_data
    # length, channels = t.shape[0], quantiles.shape[-1]
    # training_data_lambdas = torch.zeros_like(training_data)

    multi_indices = list(product(*list(map(range, quantiles.shape[:-1]))))
    splines = [None for _ in range(len(list(multi_indices)))]
    runner = 0
    for idx in multi_indices:
        curr = [slice(i, i + 1) for i in idx]
        quant_idx = curr + [slice(None)]
        coeffs = natural_cubic_spline_coeffs(
            t, quantiles[quant_idx].squeeze().unsqueeze(-1)
        )
        splines[runner] = NaturalCubicSpline(coeffs)
        runner += 1
        if runner % 100 == 0:
            print(f"Finished {runner} splines")

    random_idx = [np.random.randint(0, i) for i in quantiles.shape[:-1]]
    # random_slice = [slice(i, i + 1) for i in random_idx] + [slice(None)]
    runner = np.prod(random_idx)

    iter = 0
    max_iter = 10
    for idx in multi_indices:
        num_probs = len(p)
        probs = np.random.rand(num_probs - 2)
        probs = np.insert(probs, [0, -1], [0.0, 1.0])
        probs = np.sort(probs)
        probs = torch.tensor(probs, dtype=torch.float32)
        curr_slice = [slice(i, i + 1) for i in idx]
        quant_idx = curr_slice + [slice(None)]
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 4, 1)
        plt.plot(
            p,
            quantiles[quant_idx].reshape(-1),
            label='True Quantile',
            color='blue',
        )
        plt.plot(
            probs,
            splines[iter].evaluate(probs).reshape(-1),
            label='Spline Approximation',
            linestyle='--',
            color='red',
        )
        plt.legend()
        plt.title("Quantile spline approx")

        plt.subplot(1, 4, 2)
        plt.plot(t, training_data[quant_idx].reshape(-1), label='Raw Data')
        plt.plot(t, renorm_data[quant_idx].reshape(-1), label='Renorm Data')
        plt.legend()
        plt.title("Raw Data")

        plt.subplot(1, 4, 3)
        plt.plot(t, cdf[quant_idx].reshape(-1), label='True CDF')
        plt.title("CDF")

        plt.subplot(1, 4, 4)
        sampling_disc = p - probs
        plt.plot(range(len(sampling_disc)), sampling_disc)
        plt.title("Sampling Discrepancy")
        plt.savefig(f'quantile_spline{iter}.jpg')
        plt.close()

        print(f"Finished {iter} plots")
        iter += 1

        if iter >= max_iter:
            break


if __name__ == "__main__":
    main()
