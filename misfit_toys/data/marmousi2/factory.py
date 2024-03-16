import os
from warnings import warn

import deepwave as dw
import torch
from deepwave.common import vpvsrho_to_lambmubuoyancy as get_lame

# add_root_package_path(path=os.path.dirname(__file__), pkg="misfit_toys")
from mh.core import DotDict

from misfit_toys.data.dataset import DataFactory, fixed_rec, towed_src


class Factory(DataFactory):
    def _manufacture_data(self):
        if self.installed(
            "vp_true",
            "vs_true",
            "rho_true",
            "src_loc_y",
            "rec_loc_y",
            "src_amp_y",
        ):
            return

        d = DotDict(self.process_web_data())

        self.tensors.vp_true = d.vp_true
        self.tensors.vs_true = d.vs_true
        self.tensors.rho_true = d.rho_true

        d.ny, d.nx = self.tensors.vp_true.shape

        self.tensors.src_loc_y = towed_src(
            n_shots=d.n_shots,
            src_per_shot=d.src_per_shot,
            d_src=d.d_src,
            fst_src=d.fst_src,
            src_depth=d.src_depth,
            d_intra_shot=d.d_intra_shot,
        )

        self.tensors.rec_loc_y = fixed_rec(
            n_shots=d.n_shots,
            rec_per_shot=d.rec_per_shot,
            d_rec=d.d_rec,
            fst_rec=d.fst_rec,
            rec_depth=d.rec_depth,
        )

        # source_amplitudes
        self.tensors.src_amp_y = dw.wavelets.ricker(
            d.freq, d.nt, d.dt, d.peak_time
        ).repeat(d.n_shots, d.src_per_shot, 1)

        return d


def main():
    f = Factory.cli_construct(
        device="cuda:0", src_path=os.path.dirname(__file__)
    )
    f.manufacture_data()


if __name__ == "__main__":
    main()
