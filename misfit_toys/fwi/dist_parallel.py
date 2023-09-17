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
from ..utils import *
from .modules.seismic_data import SeismicData

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

def setup_distribution(
    *,
    obs_data,
    src_amp,
    src_loc,
    rec_loc,
    model,
    dx,
    dt,
    freq,
    rank,
    world_size
):
    #chunk the data according to rank
    obs_data = torch.chunk(
        obs_data, 
        world_size
    )[rank].to(rank)

    src_amp = torch.chunk(
        src_amp, 
        world_size
    )[rank].to(rank)

    src_loc = torch.chunk(
        src_loc, 
        world_size
    )[rank].to(rank)

    rec_loc = torch.chunk(
        rec_loc, 
        world_size
    )[rank].to(rank)

    prop = Prop(model, dx, dt, freq).to(rank)
    prop = DDP(prop, device_ids=[rank])
    return prop, obs_data, src_amp, src_loc, rec_loc

def make_plots(*, v_true, v_init, model):
    v = model()
    vmin = v_true.min()
    vmax = v_true.max()
    _, ax = plt.subplots(3, figsize=(10.5, 10.5), sharex=True,
                        sharey=True)
    ax[0].imshow(v_init.cpu().T, aspect='auto', cmap='gray',
                vmin=vmin, vmax=vmax)
    ax[0].set_title("Initial")
    ax[1].imshow(v.detach().cpu().T, aspect='auto', cmap='gray',
                vmin=vmin, vmax=vmax)
    ax[1].set_title("Out")
    ax[2].imshow(v_true.cpu().T, aspect='auto', cmap='gray',
                vmin=vmin, vmax=vmax)
    ax[2].set_title("True")
    plt.tight_layout()
    plt.savefig('example_distributed_ddp.jpg')

    v.detach().cpu().numpy().tofile('marmousi_v_inv.bin')

def train(*, prop, src_amp, src_loc, rec_loc, obs_data, dt, rank):
    # Setup optimiser to perform inversion
    loss_fn = torch.nn.MSELoss()

    # Run optimisation/inversion
    n_epochs = 2

    for cutoff_freq in [10, 15, 20, 25, 30]:
        sos = butter(6, cutoff_freq, fs=1/dt, output='sos')
        sos = [torch.tensor(sosi).to(obs_data.dtype).to(rank) for sosi in sos]

        def filt(x):
            return biquad(biquad(biquad(x, *sos[0]), *sos[1]),
                          *sos[2])
        observed_data_filt = filt(obs_data)
        optimiser = torch.optim.LBFGS(prop.parameters())
        for epoch in range(n_epochs):
            def closure():
                optimiser.zero_grad()
                out = prop(src_amp, src_loc, rec_loc)
                out_filt = filt(taper(out[-1], 100))
                loss = 1e6*loss_fn(out_filt, observed_data_filt)
                print(
                    f'Rank={rank}, Freq={cutoff_freq}, Epoch={epoch}, ' +
                    f'Loss={loss.item():.4e}',
                    flush=True
                )
                loss.backward()
                return loss

            optimiser.step(closure)

def run_rank(rank, world_size):
    print(f"Running DDP on rank {rank} / {world_size}.")
    setup(rank, world_size)
    ny, nx, nt = 2301, 751, 300
    dy, dx, dt = 4.0, 4.0, 0.004

    n_shots, src_per_shot, rec_per_shot = 16, 1, 100

    freq = 25
    peak_time = 1.5 / freq

    data = SeismicData()

    #source locations
    src_loc = towed_src(
        n_shots=n_shots,
        src_per_shot=src_per_shot,
        d_src=20,
        fst_src=10,
        src_depth=2,
        d_intra_shot=0
    )

    #receiver locations
    rec_loc = fixed_rec(
        n_shots=n_shots,
        rec_per_shot=rec_per_shot,
        d_rec=6,
        rec_depth=2,
        fst_rec=0
    )

    # source amplitudes
    src_amp = (
        (dw.wavelets.ricker(freq, nt, dt, peak_time))
        .repeat(n_shots, src_per_shot, 1)
    )

    model = Model(data.v_init, 1000, 2500)

    #Setup distribution onto multiple GPUs
    prop, data.obs_data, src_amp, src_loc, rec_loc = setup_distribution(
        obs_data=data.obs_data,
        src_amp=src_amp,
        src_loc=src_loc,
        rec_loc=rec_loc,
        model=model,
        dx=dx,
        dt=dt,
        freq=freq,
        rank=rank,
        world_size=world_size
    )

    #Perform training
    train(
        prop=prop,
        src_amp=src_amp,
        src_loc=src_loc,
        rec_loc=rec_loc,
        obs_data=data.obs_data,
        dt=dt,
        rank=rank
    )

    # Plot
    if rank == 0:
        make_plots(v_true=data.v_true, v_init=data.v_init, model=model)

    cleanup()


def run(world_size):

    mp.spawn(run_rank,
             args=(world_size,),
             nprocs=world_size,
             join=True)

def main():
    n_gpus = torch.cuda.device_count()
    run(n_gpus)

if __name__ == "__main__":
    main()