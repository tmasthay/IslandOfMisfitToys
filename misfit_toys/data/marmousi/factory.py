# from ...utils import DotDict
# from ..dataset import DataFactory, towed_src, fixed_rec
import os
import torch
from warnings import warn
import deepwave as dw
from scipy.ndimage import gaussian_filter
import copy
import sys
from masthay_helpers.global_helpers import add_root_package_path

add_root_package_path(path=os.path.dirname(__file__), pkg='misfit_toys')
from misfit_toys.data.dataset import DataFactory, towed_src, fixed_rec
from misfit_toys.utils import DotDict


class Factory(DataFactory):
    def _manufacture_data(self):
        if self.installed(
            'vp_true',
            'rho_true',
            'src_loc_y',
            'rec_loc_y',
            'obs_data',
        ):
            return

        d = DotDict(self.process_web_data())

        self.tensors.vp_init = torch.tensor(
            1 / gaussian_filter(1 / d.vp_true.cpu().numpy(), 40)
        )
        self.tensors.vp = d.vp_true.to(self.device)

        d.ny, d.nx = self.tensors.vp.shape

        self.tensors.src_loc_y = towed_src(
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

        print(f'Building obs_data in {self.out_path}...', end='', flush=True)
        self.tensors.obs_data = dw.scalar(
            self.tensors.vp,
            d.dy,
            d.dt,
            source_amplitudes=self.tensors.src_amp_y,
            source_locations=self.tensors.src_loc_y,
            receiver_locations=self.tensors.rec_loc_y,
            pml_freq=d.freq,
            accuracy=d.accuracy,
        )[-1]
        print('SUCCESS', flush=True)


def main():
    f = Factory.cli_construct(
        device='cuda:0', src_path=os.path.dirname(__file__)
    )
    f.manufacture_data()


if __name__ == "__main__":
    main()
