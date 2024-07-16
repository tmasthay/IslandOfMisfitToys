# from ...utils import DotDict
# from ..dataset import DataFactory, towed_src, fixed_rec
import os
from os.path import join as pj

import deepwave as dw
import torch

# add_root_package_path(path=os.path.dirname(__file__), pkg="misfit_toys")
from mh.core import DotDictImmutable as DDI
from scipy.ndimage import gaussian_filter

from misfit_toys.data.dataset import DataFactory, fixed_rec, towed_src


class Factory(DataFactory):
    def _manufacture_data(self):
        if self.installed("vp_true", "src_loc_y", "rec_loc_y", "obs_data"):
            return

        meta = DDI(self.metadata)
        self.tensors.vp_true = torch.load(pj(self.src_path, "vstr2A.pt"))
        self.tensors.vp_init = torch.tensor(
            1 / gaussian_filter(1 / self.tensors.vp_true.numpy(), 40)
        )
        self.tensors.vp = self.tensors.vp_true.to(self.device)

        self.tensors.src_loc_y = towed_src(
            n_shots=meta.n_shots,
            src_per_shot=meta.src_per_shot,
            d_src=meta.d_src,
            fst_src=meta.fst_src,
            src_depth=meta.src_depth,
            d_intra_shot=meta.d_intra_shot,
        ).to(self.device)

        self.tensors.rec_loc_y = fixed_rec(
            n_shots=meta.n_shots,
            rec_per_shot=meta.rec_per_shot,
            d_rec=meta.d_rec,
            fst_rec=meta.fst_rec,
            rec_depth=meta.rec_depth,
        ).to(self.device)

        self.tensors.src_amp_y = (
            dw.wavelets.ricker(meta.freq, meta.nt, meta.dt, meta.peak_time)
            .repeat(meta.n_shots, meta.src_per_shot, 1)
            .to(self.device)
        )

        print(
            f"Building obs_data for SIGSBEE-A in {self.out_path}...",
            end="",
            flush=True,
        )
        self.tensors.obs_data = dw.scalar(
            self.tensors.vp,
            meta.dy,
            meta.dt,
            source_amplitudes=self.tensors.src_amp_y,
            source_locations=self.tensors.src_loc_y,
            receiver_locations=self.tensors.rec_loc_y,
            pml_freq=meta.freq,
            accuracy=meta.accuracy,
        )[-1]
        print("SUCCESS", flush=True)


def main():
    f = Factory.cli_construct(device=None, src_path=os.path.dirname(__file__))
    f.manufacture_data()


if __name__ == "__main__":
    main()
