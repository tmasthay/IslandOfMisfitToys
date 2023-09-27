from ..dataset import towed_src, fixed_rec, DataFactory
from ...utils import DotDict
from .metadata import metadata
import os
import torch
from warnings import warn
import deepwave as dw


class Factory(DataFactory):
    def _manufacture_data(self):
        d = DotDict(self.process_web_data())
        self.tensors.vp = torch.load(os.path.join(self.src_path, 'vp.pt'))
        d.ny, d.nx = self.tensors.vp.shape

        self.src_loc_y = towed_src(
            n_shots=d.n_shots,
            src_per_shot=d.src_per_shot,
            d_src=d.d_src,
            fst_src=d.fst_src,
            src_depth=d.src_depth,
            d_intra_shot=d.d_intra_shot,
        ).to(self.device)

        self.tensors.rec_loc_y = fixed_rec(
            n_shots=d.n_shots,
            rec_per_shot=d.rec_per_shot,
            d_rec=d.d_rec,
            fst_rec=d.fst_rec,
            rec_depth=d.rec_depth,
        ).to(self.device)

        # source_amplitudes
        self.tensors.src_amp_y = (
            dw.wavelets.ricker(d.freq, d.nt, d.dt, d.peak_time)
            .repeat(d.n_shots, d.src_per_shot, 1)
            .to(self.device)
        )

        print('Building obs_data')

        # TODO: Update to elastic case
        self.tensors.out = dw.scalar(
            self.tensors.vp,
            d.dx,
            d.dt,
            source_amplitudes=self.tensors.src_amp,
            source_locations=self.tensors.src_loc,
            receiver_locations=self.tensors.rec_loc_y,
            pml_freq=d.freq,
            accuracy=d.accuracy,
        )[-1].to('cpu')

        return d
