from misfit_toys.data.dataset import field_getter, field_saver, get_data3, fetch_warn

import os
import torch
from torchaudio.functional import biquad
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from scipy.ndimage import gaussian_filter
from scipy.signal import butter
import matplotlib.pyplot as plt
import deepwave
from deepwave import scalar
from warnings import warn
from itertools import product as prod


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


# Generate a velocity model constrained to be within a desired range
class Model(torch.nn.Module):
    def __init__(self, initial, min_vel, max_vel):
        super().__init__()
        self.min_vel = min_vel
        self.max_vel = max_vel
        self.model = torch.nn.Parameter(
            torch.logit((initial - min_vel) /
                        (max_vel - min_vel))
        )

    def forward(self):
        return (torch.sigmoid(self.model) *
                (self.max_vel - self.min_vel) +
                self.min_vel)


class Prop(torch.nn.Module):
    def __init__(self, model, dx, dt, freq):
        super().__init__()
        self.model = model
        self.dx = dx
        self.dt = dt
        self.freq = freq

    def forward(self, source_amplitudes, source_locations,
                receiver_locations):
        v = self.model()
        return scalar(
            v, self.dx, self.dt,
            source_amplitudes=source_amplitudes,
            source_locations=source_locations,
            receiver_locations=receiver_locations,
            max_vel=2500,
            pml_freq=self.freq,
            time_pad_frac=0.2,
        )

def generate_data(rank, world_size):
    print(f"Running DDP on rank {rank} / {world_size}.")
    setup(rank, world_size)
    ny = 2301
    nx = 751
    dx = 4.0

    has_vp_bin = True
    try:
        v_true = torch.from_file('marmousi_vp.bin',
                             size=ny*nx).reshape(ny, nx)
    except:
        fetch_warn()
        has_vp_bin = False
        get = field_getter('conda/data/marmousi')
        v_true = get('vp_true')

    # Select portion of model for inversion
    ny = 600
    nx = 250
    v_true = v_true[:ny, :nx]

    # Smooth to use as starting model
    v_init = torch.tensor(1/gaussian_filter(1/v_true.numpy(), 40))

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

    try:
        observed_data = (
            torch.from_file('marmousi_data.bin',
                            size=n_shots*n_receivers_per_shot*nt)
            .reshape(n_shots, n_receivers_per_shot, nt)
        )
    except:
        if( has_vp_bin ):
            raise ValueError(
                'See code...has should be impossible '
                    'when trying to fetch observed_data'
            )
        fetch_warn()
        observed_data = get('obs_data')

    def taper(x):
        # Taper the ends of traces
        return deepwave.common.cosine_taper_end(x, 100)

    # Select portion of data for inversion
    n_shots = 16
    n_receivers_per_shot = 100
    nt = 300
    observed_data = (
        taper(observed_data[:n_shots, :n_receivers_per_shot, :nt])
    )

    # source_locations
    source_locations = torch.zeros(n_shots, n_sources_per_shot, 2,
                                   dtype=torch.long)
    source_locations[..., 1] = source_depth
    source_locations[:, 0, 0] = (torch.arange(n_shots) * d_source +
                                 first_source)

    # receiver_locations
    receiver_locations = torch.zeros(n_shots, n_receivers_per_shot,
                                     2, dtype=torch.long)
    receiver_locations[..., 1] = receiver_depth
    receiver_locations[:, :, 0] = (
        (torch.arange(n_receivers_per_shot) * d_receiver +
         first_receiver)
        .repeat(n_shots, 1)
    )

    # source_amplitudes
    source_amplitudes = (
        (deepwave.wavelets.ricker(freq, nt, dt, peak_time))
        .repeat(n_shots, n_sources_per_shot, 1)
    )

    observed_data = \
        torch.chunk(observed_data, world_size)[rank].to(rank)
    source_amplitudes = \
        torch.chunk(source_amplitudes, world_size)[rank].to(rank)
    source_locations = \
        torch.chunk(source_locations, world_size)[rank].to(rank)
    receiver_locations = \
        torch.chunk(receiver_locations, world_size)[rank].to(rank)

    model = Model(v_init, 1000, 2500)
    prop = Prop(model, dx, dt, freq).to(rank)
    prop = DDP(prop, device_ids=[rank])

    # Setup optimiser to perform inversion
    loss_fn = torch.nn.MSELoss()

    # Run optimisation/inversion
    n_epochs = 2

    freqs = [10, 15, 20, 25, 30]

    vp_record = torch.zeros(len(freqs), n_epochs, *v_true.shape)
    print(f'TRAIN BEGIN, Rank={rank}', flush=True)
    for (idx, cutoff_freq) in enumerate(freqs):
        sos = butter(6, cutoff_freq, fs=1/dt, output='sos')
        sos = [torch.tensor(sosi).to(observed_data.dtype).to(rank)
               for sosi in sos]

        def filt(x):
            return biquad(biquad(biquad(x, *sos[0]), *sos[1]),
                          *sos[2])
        observed_data_filt = filt(observed_data)
        optimiser = torch.optim.LBFGS(prop.parameters())
        epoch_loss = 0.0
        closure_calls = 0
        for epoch in range(n_epochs):
            def closure():
                nonlocal closure_calls, epoch_loss
                closure_calls += 1
                optimiser.zero_grad()
                out = prop(source_amplitudes, source_locations,
                           receiver_locations)
                out_filt = filt(taper(out[-1]))
                loss = 1e6*loss_fn(out_filt, observed_data_filt)
                if( closure_calls == 1 ):
                    epoch_loss = loss.item()
                loss.backward()
                return loss

            optimiser.step(closure)
            vp_record[idx, epoch] = model().detach().cpu()
            print(
                f'Loss={epoch_loss:.16f}, '
                    f'Freq={cutoff_freq}, '
                    f'Epoch={epoch}, '
                    f'Rank={rank}',
                flush=True
            )
    print(f'TRAIN END, Rank={rank}', flush=True)
    torch.save(torch.Tensor(freqs), f'deepwave/data/freqs.pt')

def run_rank(rank, world_size):
    setup(rank, world_size)
    data_save = 'deepwave/data'
    fig_save = 'deepwave/figs'

    output_files = [
        f'{data_save}/{e}.pt' 
            for e in ['vp_record', 'vp_init', 'vp_true', 'freqs']
    ]
    if( not all([os.path.exists(f) for f in output_files]) ):
        generate_data(rank, world_size)
    else:
        if( rank == 0 ):
            print(
                'Skipping data generation, delete pt files in {data_save} '
                    'and re-run this script to regenerate', 
                flush=True
            )
    
    torch.distributed.barrier()
    print(f'LOAD BEGIN, Path={data_save}/vp_record.pt, Rank={rank}', flush=True)
    vp_record = torch.load(f'{data_save}/vp_record.pt')
    v_init = torch.load(f'{data_save}/vp_init.pt')
    v_true = torch.load(f'{data_save}/vp_true.pt')
    freqs = torch.load(f'{data_save}/freqs.pt')
    
    # Plot
    if rank == 0:
        torch.save(vp_record, f'{data_save}/vp_record.pt')
        torch.save(v_init, f'{data_save}/vp_init.pt')
        torch.save(v_true, f'{data_save}/vp_true.pt')
 
        v = vp_record[-1, -1, :, :]
        vmin = v_true.min()
        vmax = v_true.max()
        cmap = 'gray'
        _, ax = plt.subplots(3, figsize=(10.5, 10.5), sharex=True,
                             sharey=True)
        ax[0].imshow(v.T, aspect='auto', cmap=cmap,
                     vmin=vmin, vmax=vmax)
        ax[0].set_title("Initial")
        ax[1].imshow(v.T, aspect='auto', cmap=cmap,
                     vmin=vmin, vmax=vmax)
        ax[1].set_title("Out")
        ax[2].imshow(v_true.T, aspect='auto', cmap=cmap,
                     vmin=vmin, vmax=vmax)
        ax[2].set_title("True")
        plt.tight_layout()
        plt.savefig(f'{fig_save}/deepwave_ddp_final_inversion.jpg')

        for freq_idx,epoch in prod(
            range(vp_record.shape[0]), 
            range(vp_record.shape[1])
        ):
            ax[1].imshow(vp_record[freq_idx,epoch].cpu().T, aspect='auto', cmap=cmap,
                       vmin=vmin, vmax=vmax)
            ax[1].set_title(f"Epoch {epoch}, cutoff freq {freqs[freq_idx]}")
            plt.savefig(f'{fig_save}/deepwave_ddp_{freq_idx}_{epoch}.jpg')
        
        dw_iters = f'{fig_save}/deepwave_ddp_[0-9]*_[0-9]*.jpg'
        cmd = (
            f'convert -delay 100 -loop 0 '
                f'$(ls -tr {dw_iters}) {fig_save}/deepwave_ddp.gif'
        )
        print('GIF creation attempt')
        print(f'    cmd="{cmd}"')
        os.system(cmd)
        os.system(f'rm {dw_iters}')

        # v.detach().cpu().numpy().tofile('marmousi_v_inv.bin')
    cleanup()


def run(world_size):

    mp.spawn(run_rank,
             args=(world_size,),
             nprocs=world_size,
             join=True)


if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    run(n_gpus)