import torch
import torch.autograd as autograd
import torch.nn as nn

from misfit_toys.fwi.loss.w2 import cts_quantile, cum_trap, unbatch_spline_eval


# Define the custom autograd function for MSE Loss
class WassersteinFunction(autograd.Function):
    @staticmethod
    def forward(ctx, f, t, renorm, quantiles):
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
        return res

    @staticmethod
    def backward(ctx, grad_output):
        t_expand, f_tilde, F, diff = ctx.saved_tensors
        quantiles = ctx.quantiles
        integrand = (
            -2 * diff * f_tilde * unbatch_spline_eval(quantiles, F, deriv=True)
        )
        integral = cum_trap(
            integrand, dx=t_expand[0, 0, 1] - t_expand[0, 0, 0], dim=-1
        )
        integral = integral.flip(dims=(-1,))
        grad_input = integral + diff**2
        return grad_input * grad_output, None, None, None


# Define the MSE loss module using the custom function
class WassersteinLoss(nn.Module):
    def __init__(self, *, t, renorm, obs_data, p):
        super().__init__()
        self.obs_data = obs_data
        self.device = obs_data.device
        self.t = t.to(self.device)
        self.renorm = renorm
        self.renorm_obs_data = renorm(obs_data).to(self.device)
        self.quantiles = cts_quantile(self.renorm_obs_data, t, p)

    def forward(self, input):
        return WassersteinFunction.apply(
            input, self.t, self.renorm, self.quantiles
        )


# Example usage
input = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
target = torch.tensor([0.0, 1.0, 2.0])

# Simple hidden layer that squares each element
squared_input = input**2
squared_input.retain_grad()

# Create an instance of the custom MSE loss
# mse_loss = MSELoss()
mse_loss = None

# Calculate loss
loss = mse_loss(squared_input, target)
print(f"Loss: {loss.item()}")

# Calculate gradients
loss.backward()
print(f"Gradient w.r.t. input: {input.grad}")
print(f"Gradient w.r.t. final layer: {squared_input.grad}")
print(
    "\nExpected gradient w.r.t. input:"
    f" {4 / 3 * input * (squared_input - target)}"
)
print(
    f"Expected gradient w.r.t. final layer: {2 / 3 * (squared_input - target)}"
)
