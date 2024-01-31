import os

import matplotlib.pyplot as plt
import torch
import torch.autograd as autograd
import torch.nn as nn
from scipy.ndimage import median_filter, uniform_filter
from torchcubicspline import natural_cubic_spline_coeffs

from misfit_toys.fwi.loss.w2 import (
    cts_quantile,
    cum_trap,
    unbatch_spline_eval,
    unbatch_splines,
)
from misfit_toys.utils import tensor_summary


# Define the custom autograd function for MSE Loss
class WassersteinFunction(autograd.Function):
    @staticmethod
    def forward(ctx, f, t, renorm, quantiles, q_deriv):
        f_tilde = renorm(f)
        t_expand = t.expand(*quantiles.shape, -1)
        F = cum_trap(f_tilde, dx=t[1] - t[0]).to(f.device)
        off_diag = unbatch_spline_eval(quantiles, F)
        diff = t_expand - off_diag
        integrated = torch.trapezoid(
            diff**2 * f_tilde, dx=t[1] - t[0], dim=-1
        )
        res = integrated.sum()
        ctx.save_for_backward(t_expand, f_tilde, F, diff)
        ctx.quantiles = quantiles
        ctx.q_deriv = q_deriv
        return res

    @staticmethod
    def backward(ctx, grad_output):
        t_expand, f_tilde, F, diff = ctx.saved_tensors
        # quantiles = ctx.quantiles
        q_deriv = ctx.q_deriv
        integrand = -2 * diff * f_tilde * unbatch_spline_eval(q_deriv, F)
        integral = cum_trap(
            integrand, dx=t_expand[0, 0, 1] - t_expand[0, 0, 0], dim=-1
        )
        integral_flipped = (
            integral[..., -1].expand(*integral.shape, -1) - integral
        )
        grad_input = integral_flipped + diff**2
        return grad_input * grad_output, None, None, None, None


# Define the MSE loss module using the custom function
class WassersteinLoss(nn.Module):
    def __init__(self, *, t, renorm, obs_data, p, filter_func):
        super().__init__()
        self.obs_data = obs_data
        self.device = obs_data.device
        self.t = t.to(self.device)
        self.renorm = renorm
        self.renorm_obs_data = renorm(obs_data).to(self.device)
        self.quantiles = cts_quantile(self.renorm_obs_data, t, p)
        print(self.quantiles.shape)
        d1 = unbatch_spline_eval(self.quantiles, p, deriv=True)
        d1 = filter_func(d1)
        coeffs = natural_cubic_spline_coeffs(p, d1.unsqueeze(-1))
        self.q_deriv = unbatch_splines(coeffs)

    def forward(self, input):
        return WassersteinFunction.apply(
            input, self.t, self.renorm, self.quantiles, self.q_deriv
        )


class W2Loss(nn.Module):
    def __init__(self, *, t, renorm, obs_data, p):
        super().__init__()
        self.obs_data = obs_data
        self.device = obs_data.device
        self.t = t.to(self.device)
        self.renorm = renorm
        self.renorm_obs_data = renorm(obs_data).to(self.device)
        self.quantiles = cts_quantile(self.renorm_obs_data, t, p)
        self.t_expand = t.expand(*self.quantiles.shape, -1)

    def batch_forward(self, f):
        f_tilde = self.renorm(f)
        F = cum_trap(f_tilde, dx=self.t[1] - self.t[0]).to(self.device)
        off_diag = unbatch_spline_eval(self.quantiles, F)
        diff = (self.t_expand - off_diag) ** 2

        integrated = torch.trapezoid(
            diff * f_tilde, dx=self.t[1] - self.t[0], dim=-1
        )
        # trace_by_trace = integrated.sum()
        # return trace_by_trace
        return integrated

    def forward(self, f):
        return self.batch_forward(f).sum()


def get_full(loss_function, inp, t, renorm, obs_data, p, filter_func):
    inp2 = inp.clone()
    inp2.requires_grad = True
    loss = loss_function(
        t=t, renorm=renorm, obs_data=obs_data, p=p, filter_func=filter_func
    )
    output = loss(inp2)
    output.backward()
    grad = inp2.grad
    return output, grad, loss


def get(loss_function):
    def filter_func(x):
        return torch.from_numpy(
            uniform_filter(x.detach().cpu().numpy(), size=5)
        ).to(x.device)

    return get_full(loss_function, inp2, t, lambda x: x, target, p, filter_func)


def test_uniform(output, grad, a, b, c, d, name):
    R = (d - c) / (b - a)
    mu = 1 - R
    beta = R * a - c
    if abs(R - 1.0) < 1e-3:
        true_loss = beta**2
        true_grad = (a - b) ** 2 - 2 * (a - c) * (b - t)
    else:
        true_loss = (
            1
            / (3.0 * mu * (b - a))
            * ((mu * b + beta) ** 3 - (mu * a + beta) ** 3)
        )
        true_grad = (mu * t + beta) ** 2 - R / mu * (
            (mu * b + beta) ** 3 - (mu * t + beta) ** 3
        )
    print('Uniform')
    print('    Computed loss: ', output.item())
    print('    True loss:     ', true_loss)
    print('    Error:         ', abs(output.item() - true_loss))
    if true_loss > 0:
        print(
            '    Relative error:',
            abs(output.item() - true_loss) / true_loss,
        )
    else:
        print('    Relative error: N/A...zero loss')
    print('\nUniform Gradient')
    print('    Computed grad: ', grad.squeeze())
    print('    True grad:     ', true_grad)
    print('    L_inf Error:         ', abs(grad.squeeze() - true_grad).max())
    print(
        '    L_inf Relative error:',
        (abs(grad.squeeze() - true_grad) / true_grad).max(),
    )
    print(
        '    MSE:         ',
        abs(grad.squeeze() - true_grad).norm() / grad.numel(),
    )
    print(
        '    MSE Relative:    ',
        abs(grad.squeeze() - true_grad).norm() / true_grad.norm(),
    )
    plt.clf()
    plt.plot(t, grad.squeeze(), label='computed')
    plt.plot(t, true_grad, label='true')
    plt.title('Gradient: ' + name)
    plt.legend()
    plt.savefig(name + '_grad.jpg')
    plt.close()


test = 'uniform'

if test == 'gaussian':
    N = 1000
    T = 20
    sig1, sig2 = 1.0, 5.0
    mu1, mu2 = 1.0, 2.0
    t = torch.linspace(-T, T, N)
    p = torch.linspace(0, 1, N)
    inp2 = torch.exp(-0.5 * ((t - mu1) / sig1) ** 2)
    target = torch.exp(-0.5 * ((t - mu2) / sig2) ** 2)
    inp2 = inp2 / torch.trapz(inp2, dx=t[1] - t[0])
    target = target / torch.trapz(target, dx=t[1] - t[0])
    inp2 = inp2.expand(1, 1, -1)
    target = target.expand(1, 1, -1)
else:
    N = 10000
    a, b = 4.0, 7.0
    c, d = 4.0, 6.0
    eps = 1e-1
    t = torch.linspace(min(a, c) - 1, max(b, d) + 1, N)
    inp2 = (1.0 / (b - a)) * torch.ones(1, 1, N) * (t >= a) * (t <= b) + eps
    target = (1.0 / (d - c)) * torch.ones(1, 1, N) * (t >= c) * (t <= d) + eps
    inp2 /= torch.trapz(inp2, dx=t[1] - t[0]).detach()
    target /= torch.trapz(target, dx=t[1] - t[0]).detach()
    p = torch.linspace(0, 1, N)
    plt.clf()
    plt.plot(t, inp2.squeeze(), label=r'$f$')
    plt.plot(t, target.squeeze(), label=r'$g$')
    plt.legend()
    plt.title('PDF')
    plt.savefig('pdf.jpg')

# loss_auto, grad_auto = get(W2Loss)
loss_manual, grad_manual, obj = get(WassersteinLoss)

# test_uniform(loss_auto, grad_auto, a, b, c, d, 'auto')
test_uniform(loss_manual, grad_manual, a, b, c, d, 'manual')

p = torch.linspace(0, 1, 100)
q = obj.quantiles[0, 0]
q_deriv = obj.q_deriv[0, 0]
plt.plot(p, q.evaluate(p), label='quantile')
plt.plot(p, q_deriv.evaluate(p), label='quantile deriv (reinterpolated)')
plt.plot(p, q.derivative(p), label='quantile deriv (directly evaled)')
plt.legend()
plt.xlabel('p')
plt.ylabel('t')
plt.savefig('quantile.jpg')
plt.close()
