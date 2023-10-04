from misfit_toys.data.dataset import get_data3
from misfit_toys.fwi.modules.models import Param, ParamConstrained
from misfit_toys.fwi.modules.distribution import Distribution, setup, cleanup
from misfit_toys.utils import print_tensor, taper, get_pydict, DotDict
from misfit_toys.fwi.modules.seismic_data import SeismicProp
from misfit_toys.fwi.modules.training import Training
from misfit_toys.utils import idt_print

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
from example import Example


os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
# os.environ['TORCH_USE_CUDA_DSA'] = '1'
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def place_rank(tensor, rank, world_size):
    if tensor is None:
        return None
    elif not isinstance(tensor, torch.Tensor):
        raise TypeError(
            idt_print(
                'misfit_toys.fwi.modules.distribution.place_rank',
                f'Expected tensor, got {type(tensor)}',
                levels=1,
            )
        )
    return torch.chunk(tensor, world_size)[rank].to(rank)


class ExampleIOMT(Example):
    def _generate_data(self, rank, world_size):
        path = 'conda/data/marmousi/deepwave_example/shots16'
        meta = get_pydict(path, as_class=True)
        chunk_size = meta.n_shots // world_size
        amp_idx = torch.arange(
            rank * chunk_size, (rank + 1) * chunk_size, dtype=torch.long
        )
        prop = SeismicProp(
            path='conda/data/marmousi/deepwave_example/shots16',
            extra_forward_args={
                'max_vel': 2500,
                'time_pad_frac': 0.2,
                'pml_freq': meta.freq,
                'amp_idx': amp_idx,
            },
            vp_prmzt=ParamConstrained.delay_init(
                requires_grad=True, minv=1000, maxv=2500
            ),
            src_amp_y_prmzt=Param.delay_init(requires_grad=False),
        )
        prop.obs_data = taper(prop.obs_data, 100)
        self.tensors['vp_init'] = prop.vp().detach().cpu()
        self.tensors['vp_true'] = prop.vp_true.detach().cpu()
        self.tensors['src_amp_y'] = prop.src_amp_y.detach().cpu()
        self.tensors['rec_loc_y'] = prop.rec_loc_y.detach().cpu()
        self.tensors['obs_data'] = prop.obs_data.detach().cpu()
        self.tensors['src_loc_y'] = prop.src_loc_y.detach().cpu()

        # dstrb = Distribution(rank=rank, world_size=world_size, prop=prop)
        # print(
        #     f'BEFORE TRAIN={dstrb.dist_prop.module.src_amp_y.device}',
        #     flush=True,
        # )
        # prop.obs_data = place_rank(prop.obs_data, rank, world_size)
        # prop.src_loc_y = place_rank(prop.src_loc_y, rank, world_size)
        # prop.rec_loc_y = place_rank(prop.rec_loc_y, rank, world_size)
        # prop.src_amp_y = place_rank(prop.src_amp_y, rank, world_size)
        # prop.src_amp_x = place_rank(prop.src_amp_x, rank, world_size)

        # class Prop(torch.nn.Module):
        #     def __init__(self, model, dx, dt, freq):
        #         super().__init__()
        #         self.model = model
        #         self.dx = dx
        #         self.dt = dt
        #         self.freq = freq
        #         self.src_amp_y = prop.src_amp_y
        #         self.src_loc_y = prop.src_loc_y
        #         self.rec_loc_y = prop.rec_loc_y

        #     # def forward(
        #     #     self, source_amplitudes, source_locations, receiver_locations
        #     # ):
        #     #     v = self.model()
        #     #     return scalar(
        #     #         v,
        #     #         self.dx,
        #     #         self.dt,
        #     #         source_amplitudes=source_amplitudes,
        #     #         source_locations=source_locations,
        #     #         receiver_locations=receiver_locations,
        #     #         max_vel=2500,
        #     #         pml_freq=self.freq,
        #     #         time_pad_frac=0.2,
        #     #     )
        #     def forward(self, x):
        #         v = self.model()
        #         return scalar(
        #             v,
        #             self.dx,
        #             self.dt,
        #             source_amplitudes=self.src_amp_y,
        #             source_locations=self.src_loc_y,
        #             receiver_locations=self.rec_loc_y,
        #             max_vel=2500,
        #             pml_freq=self.freq,
        #             time_pad_frac=0.2,
        #         )

        # prop.vp.p.data = prop.vp.p.data.to(rank)
        # prop_dummy = Prop(model=prop.vp, dx=4.0, dt=0.004, freq=25)
        # prop.vp.p.data = place_rank(prop.vp.p.data, rank, world_size)

        # prop_dummy = DDP(prop, device_ids=[rank])
        # prop_dist = Distribution(rank=rank, world_size=world_size, prop=prop)

        prop = prop.chunk(rank, world_size)
        prop = prop.to(rank)
        dist_prop = DDP(prop, device_ids=[rank])
        trainer = Training(
            dist_prop=dist_prop, rank=rank, world_size=world_size
        )
        trainer.train()

        # loss_fn = torch.nn.MSELoss()
        # n_epochs = 2
        # freqs = torch.Tensor([10, 15, 20, 25, 30]).detach().cpu()
        # n_freqs = freqs.shape[0]

        # self.tensors['vp_record'] = (
        #     torch.zeros(n_freqs, n_epochs, *prop.vp().shape).detach().cpu()
        # )

        # self.tensors['loss'] = torch.zeros(n_freqs, n_epochs).detach().cpu()
        # self.tensors['freqs'] = freqs

        # loss_local = torch.zeros(freqs.shape[0], n_epochs).to(rank)
        # if rank == 0:
        #     gather_loss = [
        #         torch.zeros_like(loss_local) for _ in range(world_size)
        #     ]
        # else:
        #     gather_loss = None

        # dt = 0.004
        # tape_len = 100
        # for idx, cutoff_freq in enumerate(list(freqs)):
        #     sos = butter(6, cutoff_freq, fs=1 / dt, output='sos')
        #     sos = [
        #         torch.tensor(sosi).to(prop.obs_data.dtype).to(rank)
        #         for sosi in sos
        #     ]

        #     def filt(x):
        #         return biquad(biquad(biquad(x, *sos[0]), *sos[1]), *sos[2])

        #     observed_data_filt = filt(prop.obs_data)
        #     optimiser = torch.optim.LBFGS(prop_dummy.parameters())

        #     for epoch in range(n_epochs):
        #         epoch_loss = 0.0
        #         closure_calls = 0

        #         def closure():
        #             nonlocal closure_calls, epoch_loss
        #             closure_calls += 1
        #             optimiser.zero_grad()
        #             out = prop_dummy(1)
        #             out_filt = filt(taper(out[-1], tape_len))
        #             loss = 1e6 * loss_fn(out_filt, observed_data_filt)
        #             if closure_calls == 1:
        #                 loss_local[idx, epoch] = loss.item()
        #                 epoch_loss = loss_local[idx, epoch]
        #             loss.backward()
        #             return loss

        #         optimiser.step(closure)
        #         self.tensors['vp_record'][idx, epoch] = (
        #             prop_dummy.module.vp().detach().cpu()
        #         )
        #         print(
        #             (
        #                 f'Loss={epoch_loss:.16f}, '
        #                 f'Freq={cutoff_freq}, '
        #                 f'Epoch={epoch}, '
        #                 f'Rank={rank}'
        #             ),
        #             flush=True,
        #         )
        # self.print(f'TRAIN END, Rank={rank}')
        # self.print(loss_local)
        # torch.distributed.gather(
        #     tensor=loss_local, gather_list=gather_loss, dst=0
        # )
        # if rank == 0:
        #     self.print(f'GATHER BEGIN, Rank={rank}')
        #     self.tensors['loss'] = torch.stack(gather_loss).to('cpu')
        #     self.print(
        #         f'Gathered data: {self.tensors["loss"]} of shape'
        #         f' {self.tensors["loss"].shape}'
        #     )

    def plot_data(self, **kw):
        self.n_epochs = 2
        self.plot_inv_record_auto(
            name='vp',
            labels=[
                ('Freq', self.tensors['freqs']),
                ('Epoch', range(self.n_epochs)),
            ],
            plot_args=dict(
                transpose=True,
                vmin=self.tensors['vp_true'].min(),
                vmax=self.tensors['vp_true'].max(),
                cmap='seismic',
            ),
        )
        self.plot_loss()
        self.plot_field(field='obs_data', transpose=True, cbar='dynamic')


if __name__ == '__main__':
    iomt_example = ExampleIOMT(
        data_save='iomt/data',
        fig_save='iomt/figs',
        tensor_names=[
            'vp_true',
            'vp_record',
            'vp_init',
            'freqs',
            'loss',
            'src_amp_y',
            'obs_data',
            'rec_loc_y',
            'src_loc_y',
        ],
        verbose=2,
    )
    iomt_example.run()
