# from ...utils import DotDict
# from ..dataset import DataFactory, towed_src, fixed_rec
import copy
import os
import sys
from warnings import warn

import deepwave as dw
import torch

# add_root_package_path(path=os.path.dirname(__file__), pkg="misfit_toys")
from mh.core import DotDict
from scipy.ndimage import gaussian_filter

from misfit_toys.data.dataset import DataFactory, fixed_rec, towed_src


class Factory(DataFactory):
    def _manufacture_data(self):
        if self.installed("vp_true", "src_loc_y", "rec_loc_y", "obs_data"):
            return

        self.tensors = self.get_parent_tensors()
        d = DotDict(self.metadata)

        mean_true_vp = self.tensors.vp_true.mean()
        std_true_vp = self.tensors.vp_true.std()

        depth = self.tensors.vp_true.shape[0]
        mid_depth = depth // 2
        self.tensors.vp_true[:mid_depth] = mean_true_vp - d.beta * std_true_vp
        self.tensors.vp_true[mid_depth:] = mean_true_vp + d.beta * std_true_vp

        self.tensors.vp_init = self.tensors.vp_true.mean() * torch.ones_like(
            self.tensors.vp_true
        )
        self.tensors.vp = self.tensors.vp_true.to(self.device)

        print(f"Building obs_data in {self.out_path}...", end="", flush=True)
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
        print("SUCCESS", flush=True)


def main():
    f = Factory.cli_construct(
        device="cuda:0", src_path=os.path.dirname(__file__)
    )
    f.manufacture_data()


if __name__ == "__main__":
    main()
