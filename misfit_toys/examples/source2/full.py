import torch


def rect_grid(*, sy, ey, dy, sx, ex, dx, n_shots=None):
    for k, v in locals().items():
        if k != 'n_shots':
            assert isinstance(v, int), f'{k} must be an integer, got {type(v)}'
    assert isinstance(
        n_shots, (int, type(None))
    ), f'n_shots must be an integer or None, got {type(n_shots)}'
    y = torch.arange(sy, ey, dy)
    x = torch.arange(sx, ex, dx)
    u = torch.cartesian_prod(y, x)
    if n_shots is not None:
        return u.unsqueeze(0).expand(n_shots, *u.shape)
    else:
        return u


def cent_grid(*, cy, cx, dy=1, dx=1, ny, nx):
    for k, v in locals().items():
        assert isinstance(v, int), f'{k} must be an integer, got {type(v)}'
    y = torch.arange(cy - dy * (ny - 1) / 2, cy + dy * (ny + 1) / 2, dy)
    x = torch.arange(cx - dx * (nx - 1) / 2, cx + dx * (nx + 1) / 2, dx)
    u = torch.cartesian_prod(y, x)
    return u
