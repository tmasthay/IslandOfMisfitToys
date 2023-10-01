from misfit_toys.data.dataset import get_data3
from misfit_toys.fwi.modules.models import Param, ParamConstrained
from misfit_toys.fwi.modules.distribution import Distribution, setup, cleanup
from misfit_toys.utils import print_tensor, taper, get_pydict
from misfit_toys.fwi.modules.seismic_data import SeismicProp
from misfit_toys.fwi.modules.training import Training

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

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


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
        self.tensors['vp_init_raw'] = prop.vp().detach().cpu()
        self.tensors['vp_init'] = prop.vp().detach().cpu()
        self.tensors['vp_true'] = prop.vp_true.detach().cpu()
        self.tensors['src_amp_y'] = prop.src_amp_y.detach().cpu()
        self.tensors['rec_loc_y'] = prop.rec_loc_y.detach().cpu()
        self.tensors['obs_data'] = prop.obs_data.detach().cpu()
        self.tensors['src_loc_y'] = prop.src_loc_y.detach().cpu()
        torch.save(self.tensors['obs_data'], '/home/tyler/obs_data_iomt.pt')

        dstrb = Distribution(rank=rank, world_size=world_size, prop=prop)
        print(
            f'BEFORE TRAIN={dstrb.dist_prop.module.src_amp_y.device}',
            flush=True,
        )
        trainer = Training(distribution=dstrb)
        (
            self.tensors['loss'],
            self.tensors['freqs'],
            self.tensors['vp_record'],
        ) = trainer.train()

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
