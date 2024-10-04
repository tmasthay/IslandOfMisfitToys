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

def exc_dict(d: dict, *args) -> dict:
    return {k: d[k] for k in d.keys() if k not in args}

def gen_matrix(N, *, alpha):
    A = torch.zeros(N, N)
    v1 = torch.arange(1, N)
    v2 = pad(torch.arange(0, N - 2), (1, 0), value=0)
    v = (torch.stack([v1, v2], dim=1)) ** (1 - alpha)
    w = torch.sum(torch.tensor([1, -1])[None, :] * v, dim=1)

    for i in range(1, N):
        # input(f'{i=}, {A[i, i:].shape=}, {w[:(N-i)].shape=}, {N-i=}')
        A[i:, i] = w[: (N - i)]

    a1 = torch.arange(1, N)
    a2 = torch.arange(0, N - 1)
    a = torch.stack([a1, a2], dim=1) ** (1 - alpha)
    b = torch.sum(torch.tensor([1, -1])[None, :] * a, dim=1)

    A[1:, 0] = b

    return A


def pretty_print(A: torch.Tensor):
    assert A.ndim == 2
    for i in range(A.shape[0]):
        v = [f'{x:.2f}' for x in A[i].tolist()]
        print(' '.join(v))


def caputo(func_deriv: torch.Tensor, *, alpha: float, dx: float) -> torch.Tensor:
    assert func_deriv.ndim == 1
    assert 0 <= alpha <= 1
    assert dx > 0

    # if alpha == 0.0:
    #     return func_deriv
    if alpha == 1.0:
        # diff = (func_deriv[1:] - func_deriv[:-1]) / dx
        # return pad(diff, (1, 0), value=diff[0].item())
        return func_deriv

    N = func_deriv.shape[0]
    A = gen_matrix(N, alpha=alpha)
    # pretty_print(A)
    # input()
    C = 0.5 * dx ** (1 - alpha) / gamma(2-alpha)
    return C * torch.mv(A, func_deriv)


def preprocess_cfg(cfg: DictConfig) -> DotMap:
    return DotMap(OmegaConf.to_container(cfg, resolve=True))


@hydra.main(config_path="cfg", config_name="cfg", version_base=None)
def main(cfg: DictConfig):
    c = preprocess_cfg(cfg)
    # A = gen_matrix(c.N, alpha=c.alpha)
    # pretty_print(A)
    # input()
    # print(f'{A.norm()=}')

    x = torch.linspace(c.a, c.b, c.N)

    func = get_callback(c.f.name, **exc_dict(c.f, 'name'))

    func_evaled = func(x)
    analytic_deriv = func.deriv(x, alpha=c.alpha)
    
    spline_coeffs = natural_cubic_spline_coeffs(x, func_evaled[None, :, None])
    spline = NaturalCubicSpline(spline_coeffs)
    classical_deriv = spline.derivative(x, order=1).squeeze()
    num_deriv = caputo(classical_deriv, alpha=c.alpha, dx=x[1] - x[0])
    rel_err = torch.norm(analytic_deriv - num_deriv) / torch.norm(
        analytic_deriv
    )

    plt.subplots(3, 1, figsize=(10, 10))

    plt.subplot(3, 1, 1)
    plt.plot(x, func_evaled, label='u')
    plt.title(f'Function Evaled: {func}')

    plt.subplot(3, 1, 2)
    plt.plot(x, analytic_deriv, **c.plt.analytic)
    plt.plot(x, num_deriv, **c.plt.numerical)
    plt.title(f'{c.alpha}-th Caputo Derivative: {func}\nRel Err: {rel_err:.2e}')
    plt.legend()
    
    plt.subplot(3, 1, 3)
    plt.plot(x, classical_deriv)
    plt.title("Classical Derivative")
    
    plt.tight_layout()

    plt.savefig(hydra_out('res.jpg'))
    os.system(f'code {hydra_out("res.jpg")}')


if __name__ == "__main__":
    main()
