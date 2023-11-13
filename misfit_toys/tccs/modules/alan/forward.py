import os

import deepwave
import matplotlib.pyplot as plt
import torch
from deepwave import scalar


def main():
    def get_file(name):
        return os.path.join(os.path.dirname(__file__), 'out/base', name)

    def load(name):
        return torch.load(get_file(name))

    def save(tensor, name):
        torch.save(tensor, get_file(name))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ny = 2301
    nx = 751
    dx = 4.0
    if os.path.exists('vp.pt'):
        print('Loading vp.pt')
        v = load('vp.pt')
    else:
        print('Loading vp.bin')
        v = (
            torch.from_file(get_file('vp.bin'), size=ny * nx)
            .reshape(ny, nx)
            .to(device)
        )
        save(v.cpu(), 'vp.pt')

    n_shots = 115

    n_sources_per_shot = 1
    d_source = 20  # 20 * 4m = 80m
    first_source = 10  # 10 * 4m = 40m
    source_depth = 2  # 2 * 4m = 8m

    n_receivers_per_shot = 384
    d_receiver = 6  # 6 * 4m = 24m
    first_receiver = 0  # 0 * 4m = 0m
    receiver_depth = 2  # 2 * 4m = 8m

    freq = 25
    nt = 750
    dt = 0.004
    peak_time = 1.5 / freq

    # source_locations
    source_locations = torch.zeros(
        n_shots, n_sources_per_shot, 2, dtype=torch.long, device=device
    )
    source_locations[..., 1] = source_depth
    source_locations[:, 0, 0] = torch.arange(n_shots) * d_source + first_source

    # receiver_locations
    receiver_locations = torch.zeros(
        n_shots, n_receivers_per_shot, 2, dtype=torch.long, device=device
    )
    receiver_locations[..., 1] = receiver_depth
    receiver_locations[:, :, 0] = (
        torch.arange(n_receivers_per_shot) * d_receiver + first_receiver
    ).repeat(n_shots, 1)

    # source_amplitudes
    source_amplitudes = (
        deepwave.wavelets.ricker(freq, nt, dt, peak_time)
        .repeat(n_shots, n_sources_per_shot, 1)
        .to(device)
    )

    # Propagate
    out = scalar(
        v,
        dx,
        dt,
        source_amplitudes=source_amplitudes,
        source_locations=source_locations,
        receiver_locations=receiver_locations,
        accuracy=8,
        pml_freq=freq,
    )

    # Plot
    receiver_amplitudes = out[-1]
    vmin, vmax = torch.quantile(
        receiver_amplitudes[0], torch.tensor([0.05, 0.95]).to(device)
    )
    _, ax = plt.subplots(1, 2, figsize=(10.5, 7), sharey=True)
    ax[0].imshow(
        receiver_amplitudes[57].cpu().T,
        aspect='auto',
        cmap='gray',
        vmin=vmin,
        vmax=vmax,
    )
    ax[1].imshow(
        receiver_amplitudes[:, 192].cpu().T,
        aspect='auto',
        cmap='gray',
        vmin=vmin,
        vmax=vmax,
    )
    ax[0].set_xlabel("Channel")
    ax[0].set_ylabel("Time Sample")
    ax[1].set_xlabel("Shot")
    plt.tight_layout()
    plt.savefig('example_forward_model.jpg')

    save(receiver_amplitudes.cpu(), 'obs_data.pt')


if __name__ == '__main__':
    main()
