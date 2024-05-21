import torch
from torchcubicspline import NaturalCubicSpline, natural_cubic_spline_coeffs

if __name__ == "__main__":
    t = torch.linspace(0, 1, 7)
    # (2, 1) are batch dimensions. 7 is the time dimension
    # (of the same length as t). 3 is the channel dimension.
    x = torch.rand(1, 7, 3)
    coeffs = natural_cubic_spline_coeffs(t, x)
    # coeffs is a tuple of tensors

    # ...at this point you can save the coeffs, put them
    # through PyTorch's Datasets and DataLoaders, etc...

    spline = NaturalCubicSpline(coeffs)

    point = torch.tensor(0.4)
    # will be a tensor of shape (2, 1, 3), corresponding to
    # batch, batch, and channel dimensions
    out = spline.derivative(point)
    print(f'{point.shape=}, {out.shape=}')

    point2 = torch.rand(2, 2)
    # will be a tensor of shape (2, 1, 1, 2, 3), corresponding to
    # batch, batch, time, time and channel dimensions
    out2 = spline.derivative(point2)
    print(f'{point2.shape=}, {out2.shape=}')
