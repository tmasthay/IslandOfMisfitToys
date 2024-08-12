from typing import Union

import torch
from mh.core import enforce_types


@enforce_types
def rect_grid(
    *,
    sy: int,
    ey: int,
    dy: int,
    sx: int,
    ex: int,
    dx: int,
    n_shots: Union[int, None] = None,
    device: str = 'cpu',
):
    type_check = {
        k: (v, int)
        for k, v in locals().items()
        if k not in ['n_shots', 'device']
    }
    type_check['n_shots'] = (n_shots, (int, type(None)))
    type_check['device'] = (device, str)
    for k, v in type_check.items():
        if not isinstance(v[0], v[1]):
            raise ValueError(f"{k} must be of type {v[1]}, got {type(v[0])}")
    y = torch.arange(sy, ey, dy)
    x = torch.arange(sx, ex, dx)
    u = torch.cartesian_prod(y, x)
    if n_shots is not None:
        return u.unsqueeze(0).expand(n_shots, *u.shape).to(device)
    else:
        return u.to(device)


@enforce_types
def cent_grid(
    *,
    cy: int,
    cx: int,
    dy: int = 1,
    dx: int = 1,
    ny: int,
    nx: int,
    n_shots: Union[int, None] = None,
    device: str = 'cpu',
):
    delta_y, delta_x = dy * (ny - 1) // 2, dx * (nx - 1) // 2
    if ny != 1:
        delta_y = max(1, delta_y)
    if nx != 1:
        delta_x = max(1, delta_x)
    sy = cy - delta_y
    sx = cx - delta_x

    # ensure at least one point is included
    # NOTE: both max(1,...) are needed here...there's a cleaner way but this works for now!
    ey = cy + max(1, delta_y)
    ex = cx + max(1, delta_x)

    if ny != 1:
        ey += dy
    if nx != 1:
        ex += dx

    # input(locals())
    return rect_grid(
        sy=sy, ey=ey, dy=dy, sx=sx, ex=ex, dx=dx, n_shots=n_shots
    ).to(device)


def main_grid():
    u = rect_grid(sy=300, ey=301, dy=1, sx=300, ex=301, dx=1)
    v = cent_grid(cy=300, cx=125, dy=1, dx=2, ny=3, nx=3)

    print(u)
    print(v)


if __name__ == "__main__":
    main_grid()
