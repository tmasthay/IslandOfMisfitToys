import matplotlib.pyplot as plt
import torch
from mh.core import rand_slices
from mh.typlotlib import get_frames_bool, save_frames
from misfit_toys.utils import bool_slice
import numpy as np


def plotter(*, data, idx, fig, axes, num_samples):
    if idx[0] == 0:
        plt.clf()
    a = int(np.sqrt(num_samples))
    plt.subplot(a, a, idx[0] + 1)
    plt.plot(data['pdf'][idx], label='pdf')
    plt.plot(data['obs'][idx], label='obs')
    plt.plot(data['out'][idx], label='out')
    plt.plot(data['obs_data_org'][idx], label='obs_data_org', linestyle='--')
    plt.ylim(-1, 5)
    plt.tight_layout()
    plt.legend()

    return {'num_samples': num_samples}


def main():
    out = torch.load('out_record.pt')
    obs_data = torch.load('obs_data_renorm_record.pt')
    pdf = torch.load('pdf_record.pt')
    obs_data_org = torch.load('obs_data_record.pt')

    print(
        f'shapes\n    out: {out.shape}\n    obs_data: {obs_data.shape}\n    pdf: {pdf.shape}'
    )

    num_samples = 4
    none_dims = [0, -1]
    idx = rand_slices(*list(out.shape), none_dims=none_dims, N=num_samples)
    get = lambda x: torch.stack([x[e] for e in idx])

    out_s = get(out)
    obs_data_s = get(obs_data)
    pdf_s = get(pdf)
    obs_data_org_s = get(obs_data_org)
    print(
        f'shapes after slicing\n    out: {get(out).shape}\n    obs_data: {get(obs_data).shape}\n    pdf: {get(pdf).shape}'
    )

    def ctrl(idx, shape):
        return idx[0] == shape[0] - 1

    a = int(np.sqrt(num_samples))
    fig, axes = plt.subplots(a, a, figsize=(8, 8))
    frames = get_frames_bool(
        data={
            'out': out_s,
            'obs': obs_data_s,
            'pdf': pdf_s,
            'obs_data_org': obs_data_org_s,
        },
        iter=bool_slice(
            *out_s.shape, permute=(1, 0, 2), none_dims=[-1], ctrl=ctrl
        ),
        fig=fig,
        axes=axes,
        plotter=plotter,
        num_samples=num_samples,
    )
    save_frames(frames, path='res', movie_format='gif', duration=1000)
    print('Results saved in res.gif')


if __name__ == "__main__":
    main()
