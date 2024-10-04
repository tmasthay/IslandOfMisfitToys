import os
from matplotlib import pyplot as plt
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from dotmap import DotMap
from torch.nn.functional import pad
from functions import get_callback, gamma
from mh.core import hydra_out
from torchcubicspline import natural_cubic_spline_coeffs, NaturalCubicSpline
from misfit_toys.utils import bool_slice
from mh.typlotlib import save_frames, get_frames_bool

def exc_dict(d: dict, *args) -> dict:
    return {k: d[k] for k in d.keys() if k not in args}

def gen_matrix(N: int, *, alpha: torch.Tensor) -> torch.Tensor:
    A = torch.zeros(alpha.numel(), N, N)
    
    tmp1 = torch.arange(1, N)
    tmp2 = pad(torch.arange(0, N - 2), (1, 0), value=0)
    unscaled_diff_terms = (torch.stack([tmp1, tmp2], dim=1))
    scaled_diff_terms = unscaled_diff_terms[None, :, :] ** (1 - alpha[:, None, None])
    
    plus_minus = torch.tensor([1, -1])
    weights = torch.sum(plus_minus[None, None, :] * scaled_diff_terms, dim=-1)

    for i in range(1, N):
        # input(f'{i=}, {A[i, i:].shape=}, {w[:(N-i)].shape=}, {N-i=}')
        A[:, i:, i] = weights[:, : (N - i)]

    # build special vector for first column
    fst_col_1 = torch.arange(1, N)
    fst_col_2 = torch.arange(0, N - 1)
    unscaled_diff_fst_col = torch.stack([fst_col_1, fst_col_2], dim=1) 
    scaled_diff_fst_col = unscaled_diff_fst_col[None, :, :] ** (1 - alpha[:, None, None])
    weights_fst_col = torch.sum(plus_minus[None, None, :] * scaled_diff_fst_col, dim=-1)

    A[:, 1:, 0] = weights_fst_col
    
    for i, e in enumerate(alpha):
        if e == 1.0:
            A[i, :, :] = torch.eye(N)

    return A


def pretty_print(A: torch.Tensor):
    assert A.ndim == 2
    for i in range(A.shape[0]):
        v = [f'{x:.2f}' for x in A[i].tolist()]
        print(' '.join(v))


def caputo(func_deriv: torch.Tensor, *, alpha: torch.Tensor, dx: float) -> torch.Tensor:
    assert func_deriv.ndim == 1
    assert (0 <= alpha).all() and (alpha <= 1).all()
    assert dx > 0

    # if alpha == 0.0:
    #     return func_deriv
    # if alpha == 1.0:
    #     # diff = (func_deriv[1:] - func_deriv[:-1]) / dx
    #     # return pad(diff, (1, 0), value=diff[0].item())
    #     return func_deriv

    N = func_deriv.shape[0]
    A = gen_matrix(N, alpha=alpha)
    # pretty_print(A)
    # input()
    C = 0.5 * dx ** (1 - alpha) / gamma(2-alpha)
    return C[:, None] * torch.einsum('bij,j->bi', A, func_deriv)


def preprocess_cfg(cfg: DictConfig) -> DotMap:
    return DotMap(OmegaConf.to_container(cfg, resolve=True))


@hydra.main(config_path="cfg", config_name="cfg", version_base=None)
def main(cfg: DictConfig):
    c = preprocess_cfg(cfg)

    x = torch.linspace(c.a, c.b, c.N)
    alpha = torch.linspace(**c.alpha)
    
    func = get_callback(c.f.name, **exc_dict(c.f, 'name'))

    func_evaled = func(x)
    analytic_deriv = func.deriv(x, alpha=alpha)
    
    spline_coeffs = natural_cubic_spline_coeffs(x, func_evaled[None, :, None])
    spline = NaturalCubicSpline(spline_coeffs)
    classical_deriv = spline.derivative(x, order=1).squeeze()
    num_deriv = caputo(classical_deriv, alpha=alpha, dx=x[1] - x[0])
    rel_err = torch.norm(analytic_deriv - num_deriv, dim=1) / torch.norm(
        analytic_deriv, dim=1
    )
    
    assert num_deriv.shape == analytic_deriv.shape
    assert num_deriv.ndim == 2

    def plotter(*, data, idx, fig, axes):
        plt.clf()
        plt.plot(x, analytic_deriv[idx], **c.plt.analytic)
        plt.plot(x, num_deriv[idx], **c.plt.numerical)
        plt.plot(x, func_evaled, **c.plt.func)
        plt.plot(x, classical_deriv, **c.plt.classical)
        plt.title(f'Derivative alpha={alpha[idx[0]].item():.2f} Error={rel_err[idx[0]]:.2e}')
        plt.legend()
        
    iter = bool_slice(*num_deriv.shape, none_dims=[-1])
    frames = get_frames_bool(data=None, iter=iter, plotter=plotter)
    save_frames(frames, path=hydra_out('res'))

    os.system(f'code {hydra_out("res.gif")}')


if __name__ == "__main__":
    main()
