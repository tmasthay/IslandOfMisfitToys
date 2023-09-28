import os
import torch
from warnings import warn
import deepwave as dw
from deepwave.common import vpvsrho_to_lambmubuoyancy as get_lame
from masthay_helpers import add_root_package_path

add_root_package_path(path=os.path.dirname(__file__), pkg='misfit_toys')
from misfit_toys.data.dataset import DataFactory, towed_src, fixed_rec
from misfit_toys.utils import DotDict


class Factory(DataFactory):
    def _manufacture_data(self):
        d = DotDict(self.process_web_data())
        if d.has('obs_data'):
            print('obs_data already exists. Skipping manufacture.')
            return
        else:
            print('obs_data not found...manufacturing tensors from web data.')

        self.tensors.vp_true = d.vp_true.to(self.device)
        self.tensors.vs_true = d.vs_true.to(self.device)
        self.tensors.rho_true = d.rho_true.to(self.device)
        d.ny, d.nx = self.tensors.vp_true.shape

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

        return d


def main():
    f = Factory.cli_construct(
        device='cuda:0', src_path=os.path.dirname(__file__)
    )
    f.manufacture_data()


if __name__ == "__main__":
    main()
