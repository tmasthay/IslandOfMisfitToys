from typing import List

import deepwave as dw
import torch
from mh.core import enforce_types


def check_dim(data, *, dim, left=-torch.inf, right=torch.inf):
    val = data.shape[dim]
    if not (left <= val <= right):
        raise ValueError(
            f"Expected {left} <= val <= {right} for dim {dim}, got {val=},"
            f" {data.shape=}"
        )


def check_dims(data, *, dims, left=None, right=None):
    left = [-torch.inf] * len(dims) if left is None else left
    right = [torch.inf] * len(dims) if right is None else right
    for dim, curr_left, curr_right in zip(dims, left, right):
        check_dim(data, dim=dim, left=curr_left, right=curr_right)


def direct_load(*, path, device):
    return torch.load(path).to(device)


def build_vp(*, path, device, ny=None, nx=None):
    vp = torch.load(path).to(device)
    vp = vp.mean() * torch.ones_like(vp)
    ny = vp.shape[0] if ny is None else ny
    nx = vp.shape[1] if nx is None else nx
    return vp[:ny, :nx]


def take_first(*, path, n_shots=1, num_per_shot, device='cpu'):
    src_amp_y = torch.load(path).to(device)
    check_dims(src_amp_y, dims=[0, 1], left=[n_shots, num_per_shot])
    return src_amp_y[:n_shots, :num_per_shot, :]


def build_src_loc_y(
    *, ny, nx, n_shots, src_per_shot, rel_idx_src_loc_y, nt, dt, device
):
    if n_shots != 1 or src_per_shot != 1:
        raise NotImplementedError(
            'Only n_shots=1 and src_per_shot=1 are supported'
        )

    src_loc_y = torch.zeros(n_shots, src_per_shot, 2, dtype=torch.long)

    abs_y_idx = int(rel_idx_src_loc_y.upper_left[0] * ny)
    abs_x_idx = int(rel_idx_src_loc_y.upper_left[1] * nx)

    src_loc_y[..., 0] = abs_y_idx
    src_loc_y[..., 1] = abs_x_idx
    return src_loc_y.to(device)


def build_gbl_rec_loc(*, ny, nx, downsample_x, downsample_y, device):
    rec_loc_y = torch.zeros(ny, nx, 2, dtype=torch.long)
    rec_loc_y[..., 0] = torch.arange(ny).unsqueeze(1).repeat(1, nx)
    rec_loc_y[..., 1] = torch.arange(nx).repeat(ny, 1)
    rec_loc_y = rec_loc_y[::downsample_y, ::downsample_x]
    # flatten
    rec_loc_y = rec_loc_y.reshape(1, -1, 2)
    return rec_loc_y.to(device)


def build_gbl_obs_data(
    *,
    v,
    grid_spacing,
    dt,
    source_amplitudes,
    source_locations,
    receiver_locations,
    accuracy,
    downsample_y,
    downsample_x,
    ny,
    nx,
):
    u = dw.scalar(
        v=v,
        grid_spacing=grid_spacing,
        dt=dt,
        source_amplitudes=source_amplitudes,
        source_locations=source_locations,
        receiver_locations=receiver_locations,
        accuracy=accuracy,
    )[-1]

    shape_y = torch.ceil(torch.tensor(ny / downsample_y)).int()
    shape_x = torch.ceil(torch.tensor(nx / downsample_x)).int()
    u = u.reshape(shape_y, shape_x, -1)
    return u


def build_obs_data(
    *,
    v,
    grid_spacing,
    dt,
    source_amplitudes,
    source_locations,
    receiver_locations,
    **kw,
):
    return dw.scalar(
        v=v,
        grid_spacing=grid_spacing,
        dt=dt,
        source_amplitudes=source_amplitudes,
        source_locations=source_locations,
        receiver_locations=receiver_locations,
        **kw,
    )[-1]


def shift_data(*, data, shifts, dims=None):
    assert min(shifts) >= 0.0
    assert max(shifts) <= 1.0

    dims = list(range(len(data.shape))) if dims is None else dims
    assert len(shifts) == len(dims)
    assert len(dims) <= len(data.shape)

    abs_shifts = torch.tensor(shifts) * torch.tensor(data.shape)[dims]
    abs_shifts = abs_shifts.round().int().tolist()
    return torch.roll(data, shifts=tuple(abs_shifts), dims=dims)


# @enforce_types
def sparse_amps(
    *,
    path: str,
    n_shots: int = 1,
    num_per_shot,
    device: str = 'cpu',
    nonzeros: List[float],
):
    src_amp_y = torch.load(path)[:n_shots]
    check_dim(src_amp_y, dim=0, left=1, right=1)
    assert len(src_amp_y.shape) == 3, f'{src_amp_y.shape=}, expected 3'

    # repeat along 2nd dimension
    src_amp_y = src_amp_y.repeat(1, num_per_shot, 1)

    assert (0 <= min(nonzeros)) and (
        max(nonzeros) <= 1.0
    ), f'nonzeros must between 1 and {nonzeros=}'
    nonzeros = [int(e * src_amp_y.shape[1]) for e in nonzeros]
    for i in range(src_amp_y.shape[1]):
        if i not in nonzeros:
            src_amp_y[:, i, :] = 0.0

    # src_amp_y = src_amp_y.to_sparse().to(device)
    return src_amp_y.to(device)


if __name__ == "__main__":
    u = torch.arange(110).reshape(10, 11)
    print(shift_data(data=u, shifts=[0.0, 0.2], dims=[0, 1]))
