import deepwave as dw
import torch


def direct_load(*, path, device):
    return torch.load(path).to(device)


def build_vp(*, path, device):
    vp = torch.load(path).to(device)
    vp = vp.mean() * torch.ones_like(vp)
    return vp


def take_first(*, path, device):
    src_amp_y = torch.load(path).to(device)
    return src_amp_y[0].unsqueeze(0)


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
