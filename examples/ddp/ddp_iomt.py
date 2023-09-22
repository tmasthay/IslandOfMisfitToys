from misfit_toys.data.dataset import get_data3
from misfit_toys.fwi.modules.models import Param, ParamConstrained
from misfit_toys.fwi.modules.distribution import Distribution, setup, cleanup
from misfit_toys.utils import print_tensor, taper, get_pydict
from misfit_toys.fwi.modules.seismic_data import SeismicProp

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
from time import time

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

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

def run_rank(rank, world_size):
    path = 'conda/data/marmousi/deepwave_example/shots16'
    tape_len = 100
    meta = get_pydict(path, as_class=True)

    print(f"Running DDP on rank {rank} / {world_size}.")
    setup(rank, world_size)

    # v_true = get_data3(field='vp_true', path=path)
    # v_init = get_data3(field='vp_init', path=path)
    # observed_data = taper(
    #     get_data3(
    #         field='obs_data', 
    #         path='conda/data/marmousi/deepwave_example/shots16',
    #     ),
    #     tape_len
    # )
    # source_locations = get_data3(field='src_loc_y', path=path)
    # receiver_locations = get_data3(field='rec_loc_y', path=path)
    # source_amplitudes = Param(p=get_data3(field='src_amp_y', path=path))
    # meta = get_pydict(path)

    # observed_data = \
    #     torch.chunk(observed_data, world_size)[rank].to(rank)
    
    # source_amplitudes.p = torch.nn.Parameter(
    #     torch.chunk(source_amplitudes.p, world_size)[rank].to(rank),
    #     requires_grad=source_amplitudes.p.requires_grad
    # )
    # source_locations = \
    #     torch.chunk(source_locations, world_size)[rank].to(rank)
    # receiver_locations = \
    #     torch.chunk(receiver_locations, world_size)[rank].to(rank)

    propper = SeismicProp(
        path=path,
        extra_forward_args={ 'time_pad_frac': 0.2 },
        vp_prmzt=ParamConstrained.delay_init(
            requires_grad=True,
            minv=1000,
            maxv=2500
        )
    )
    v_init = propper.vp().detach().cpu()

    # model = ParamConstrained(
    #     p=v_init,
    #     minv=1000,
    #     maxv=2500,
    #     requires_grad=True
    # )
    # prop = Prop(model, meta.dx, meta.dt, meta.freq).to(rank)
    # prop = DDP(prop, device_ids=[rank])

    # distribution = Distribution(rank=rank, world_size=world_size, prop=propper)
    # prop = distribution.dist_prop.moduel

    propper.obs_data = torch.chunk(propper.obs_data, world_size)[rank].to(rank)
    propper.src_loc_y = torch.chunk(propper.src_loc_y, world_size)[rank].to(rank)
    propper.rec_loc_y = torch.chunk(propper.rec_loc_y, world_size)[rank].to(rank)
    propper.src_amp_y = propper.src_amp_y.to(rank)
    # propper.src_amp_y = torch.chunk(propper.src_amp_y, world_size)[rank].to(rank)

    propper.vp = propper.vp.to(rank)

    prop = DDP(propper, device_ids=[rank])

    loss_fn = torch.nn.MSELoss()

    n_epochs = 2

    begin_idx = rank * propper.src_amp_y.shape[0] // world_size
    end_idx = (rank + 1) * propper.src_amp_y.shape[0] // world_size
    amp_idx = torch.arange(begin_idx, end_idx).to(rank)

    for cutoff_freq in [10, 15, 20, 25, 30]:
        sos = butter(6, cutoff_freq, fs=1/meta.dt, output='sos')
        sos = [torch.tensor(sosi).to(prop.module.obs_data.dtype).to(rank)
               for sosi in sos]

        def filt(x):
            return biquad(biquad(biquad(x, *sos[0]), *sos[1]),
                          *sos[2])
        observed_data_filt = filt(prop.module.obs_data)
        optimiser = torch.optim.LBFGS(prop.module.parameters())

        for epoch in range(n_epochs):
            closure_calls = 0
            def closure():
                nonlocal closure_calls
                closure_calls += 1
                optimiser.zero_grad()
                out = prop(amp_idx=amp_idx)
                out_filt = filt(taper(out[-1], tape_len))
                loss = 1e6*loss_fn(out_filt, observed_data_filt)
                if( closure_calls == 1):
                    print(
                        f'Loss={loss.item():.16f}, ' 
                            f'Freq={cutoff_freq}, '
                            f'Epoch={epoch}, '
                            f'Rank={rank}',
                        flush=True
                    )
                loss.backward()

                return loss

            optimiser.step(closure)

    if rank == 0:
        v = prop.module.vp().detach().cpu()
        vmin = prop.module.vp_true.min()
        vmax = prop.module.vp_true.max()
        _, ax = plt.subplots(3, figsize=(10.5, 10.5), sharex=True,
                             sharey=True)
        ax[0].imshow(v_init.cpu().T, aspect='auto', cmap='gray',
                     vmin=vmin, vmax=vmax)
        ax[0].set_title("Initial")
        ax[1].imshow(v.T, aspect='auto', cmap='gray',
                     vmin=vmin, vmax=vmax)
        ax[1].set_title("Out")
        ax[2].imshow(prop.module.vp_true.detach().cpu().T, aspect='auto', cmap='gray',
                     vmin=vmin, vmax=vmax)
        ax[2].set_title("True")
        plt.tight_layout()
        plt.savefig('out_ddp_hybrid.jpg')

        # v.detach().cpu().numpy().tofile('marmousi_v_inv.bin')
        torch.save(v.detach().cpu(), 'marmousi_v_inv.pt')
    cleanup()


def run(world_size):

    mp.spawn(run_rank,
             args=(world_size,),
             nprocs=world_size,
             join=True)


if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    run(n_gpus)