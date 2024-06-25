"""
This module recreates the exact workflow of Alan's deepwave Marmousi benchmark example.
This script is used as an end-to-end test to make sure that our package can reproduce the results of this benchmark example.
"""

import os
import shutil

import deepwave
import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from deepwave import scalar
from scipy.ndimage import gaussian_filter
from scipy.signal import butter
from torch.nn.parallel import DistributedDataParallel as DDP
from torchaudio.functional import biquad

from misfit_toys.data.download_data import download_data
from misfit_toys.utils import get_gpu_memory, parse_path, setup, cleanup, self_read_cfg
from helpers import Prop, Model
import hydra
from omegaconf import DictConfig, OmegaConf
from mh.core import set_print_options, torch_stats

set_print_options(callback=torch_stats([
            'shape',
            'dtype',
            'min',
            'max',
        ]))

def cwd(x=''):
    if x.startswith('/'):
        return x
    return os.path.join(os.path.dirname(__file__), x)


def get_file(name, *, rank='', path='out/parallel', ext='.pt'):
    path = cwd(path)
    ext = '.' + ext.replace('.', '')
    name = name.replace(ext, '')
    if rank != '':
        rank = f'_{rank}'
    return os.path.join(os.path.dirname(__file__), path, f'{name}{rank}{ext}')


def load(name, *, rank='', path='out/parallel', ext='.pt'):
    return torch.load(get_file(name, rank=rank, path=path, ext='.pt'))


def save(tensor, name, *, rank='', path='out/parallel', ext='.pt'):
    os.makedirs(path, exist_ok=True)
    torch.save(tensor, get_file(name, rank=rank, path=path, ext='.pt'))


def savefig(name, *, path='out/parallel', ext='.pt'):
    plt.savefig(get_file(name, rank='', path=path, ext=ext))

def run_rank(rank, world_size, c):
    print(f"Running DDP on rank {rank} / {world_size}.")
    setup(rank, world_size)
    path = os.path.dirname(__file__)

    c = self_read_cfg(c, read_key='common_load')
    c = self_read_cfg(c, read_key='common_preprocess')
    print(f'{c=}')
    exit(1)


    def get(x):
        return os.path.join(path, x)

    v_true = load('vp.pt', path=get('out/base'))

    v_true = v_true[:c.ny, :c.nx]
    v_init = torch.tensor(1 / gaussian_filter(1 / v_true.numpy(), 40))

    peak_time = 1.5 / c.freq

    observed_data = load('obs_data.pt', path=get('out/base'))

    def taper(x):
        # Taper the ends of traces
        return deepwave.common.cosine_taper_end(x, 100)

    observed_data = taper(observed_data[:c.n_shots, :c.n_receivers_per_shot, :c.nt])

    # source_amplitudes
    source_amplitudes = (
        deepwave.wavelets.ricker(c.freq, c.nt, c.dt, peak_time)
    ).repeat(c.n_shots, c.n_sources_per_shot, 1)

    observed_data = torch.chunk(observed_data, world_size)[rank].to(rank)
    source_amplitudes = torch.chunk(source_amplitudes, world_size)[rank].to(
        rank
    )
    source_locations = torch.chunk(source_locations, world_size)[rank].to(rank)
    receiver_locations = torch.chunk(receiver_locations, world_size)[rank].to(
        rank
    )
    print(f'{receiver_locations.max()=}')
    print(f'{source_locations.max()=}')

    model = Model(v_init, 1000, 2500)
    prop = Prop(
        model=model,
        dx=c.dx,
        dt=c.dt,
        freq=c.freq,
        source_amplitudes=source_amplitudes,
        source_locations=source_locations,
        receiver_locations=receiver_locations
    ).to(rank)

    prop = DDP(prop, device_ids=[rank])

    # Setup optimiser to perform inversion
    loss_fn = torch.nn.MSELoss()

    # Run optimisation/inversion
    n_epochs = 2

    loss_record = []
    v_record = []
    out_record = []
    out_filt_record = []

    freqs = [10, 15, 20, 25, 30]
    # n_freqs = len(freqs)

    def get_epoch(i, j):
        return j + i * n_epochs

    for i, cutoff_freq in enumerate(freqs):
        print(i, flush=True)
        sos = butter(6, cutoff_freq, fs=1 / c.dt, output='sos')
        sos = [
            torch.tensor(sosi).to(observed_data.dtype).to(rank) for sosi in sos
        ]

        def filt(x):
            return biquad(biquad(biquad(x, *sos[0]), *sos[1]), *sos[2])

        observed_data_filt = filt(observed_data)
        optimiser = torch.optim.LBFGS(prop.parameters())
        for epoch in range(n_epochs):
            num_calls = 0

            def closure():
                nonlocal num_calls
                num_calls += 1
                optimiser.zero_grad()
                out = prop(None)
                out_filt = filt(taper(out[-1]))
                loss = 1e6 * loss_fn(out_filt, observed_data_filt)
                loss.backward()
                if num_calls == 1:
                    loss_record.append(loss.detach().cpu())
                    v_record.append(prop.module.model().detach().cpu())
                    out_record.append(out[-1].detach().cpu())
                    out_filt_record.append(out_filt.detach().cpu())
                    print(
                        f'Epoch={get_epoch(i, epoch)}, Loss={loss.item()},'
                        f' rank={rank}',
                        flush=True,
                    )
                return loss

            optimiser.step(closure)
            torch.cuda.empty_cache()

    save(torch.tensor(loss_record), 'loss_record.pt', rank=rank)
    save(torch.stack(v_record), 'vp_record.pt', rank=rank)
    save(torch.stack(out_record), 'out_record.pt', rank=rank)
    save(torch.stack(out_filt_record), 'out_filt_record.pt', rank=rank)

    torch.distributed.barrier()
    # Plot
    if rank == 0:
        loss_record = torch.mean(
            torch.stack(
                [
                    load('loss_record.pt', rank=rank)
                    for rank in range(world_size)
                ]
            ),
            dim=0,
        )
        v_record = load('vp_record.pt', rank=0)
        out_record = torch.cat(
            [load('out_record.pt', rank=rank) for rank in range(world_size)],
            dim=1,
        )
        out_filt_record = torch.cat(
            [
                load('out_filt_record.pt', rank=rank)
                for rank in range(world_size)
            ],
            dim=1,
        )

        save(loss_record, 'loss_record.pt', rank='')
        save(v_record, 'vp_record.pt', rank='')
        save(out_record, 'out_record.pt', rank='')
        save(out_filt_record, 'out_filt_record.pt', rank='')

        v = model()
        vmin = v_true.min()
        vmax = v_true.max()
        _, ax = plt.subplots(3, figsize=(10.5, 10.5), sharex=True, sharey=True)
        ax[0].imshow(
            v_init.cpu().T, aspect='auto', cmap='gray', vmin=vmin, vmax=vmax
        )
        ax[0].set_title("Initial")
        ax[1].imshow(
            v.detach().cpu().T, aspect='auto', cmap='gray', vmin=vmin, vmax=vmax
        )
        ax[1].set_title("Out")
        ax[2].imshow(
            v_true.cpu().T, aspect='auto', cmap='gray', vmin=vmin, vmax=vmax
        )
        ax[2].set_title("True")
        plt.tight_layout()
        savefig('example_distributed_ddp', ext='jpg')

        v.detach().cpu().numpy().tofile('marmousi_v_inv.bin')
    cleanup()


def run(world_size, c):
    mp.spawn(run_rank, args=(world_size,c), nprocs=world_size, join=True)

@hydra.main(config_path='cfg', config_name='cfg', version_base=None)
def main(cfg: DictConfig):
    root = os.path.abspath(os.path.dirname(__file__))
    data_path = os.path.abspath(
        os.path.join(parse_path('conda/data'), 'marmousi')
    )
    lcl_path = os.path.abspath(os.path.join(root, 'out', 'base'))
    out_path = os.path.abspath(os.path.join(root, 'out', 'parallel'))

    files = ['obs_data', 'vp']

    def all_exist(p):
        return all([os.path.exists(f'{p}/{f}.pt') for f in files])

    os.makedirs(lcl_path, exist_ok=True)
    os.makedirs(out_path, exist_ok=True)
    if not all_exist(lcl_path):
        if not all_exist(data_path):
            download_data(os.path.dirname(data_path), inclusions=['marmousi'])
        for f in files:
            shutil.copy(f'{data_path}/{f}.pt', f'{lcl_path}/{f}.pt')
    run(torch.cuda.device_count(), cfg)


if __name__ == "__main__":
    main()
