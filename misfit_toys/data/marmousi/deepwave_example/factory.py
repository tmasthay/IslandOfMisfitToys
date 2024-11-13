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


# This seems incorrect
# A totally new forward solve needs to take place!
class Factory(DataFactory):
    def _manufacture_data(self):
        if self.installed(
            "vp_true",
            "src_loc_y",
            "rec_loc_y",
            "obs_data",
            "src_amp_y",
            "rho_true",
        ):
            return

        self.tensors = self.get_parent_tensors()
        d = DotDict(self.metadata)

        v_slice, src_slice, rec_slice = DataFactory.get_slices(d)

        print("Slicing deepwave_examples tensors...", end="", flush=True)
        self.slice_subset_tensors(*v_slice, keys=["vp", "vp_true", "rho_true"])
        self.slice_subset_tensors(
            *src_slice,
            keys=["src_loc_y", "src_amp_y", "src_loc_x", "src_amp_x"],
        )
        self.slice_subset_tensors(
            *rec_slice, keys=["rec_loc_y", "rec_loc_x", "obs_data"]
        )
        self.tensors.vp_init = torch.tensor(
            1 / gaussian_filter(1 / self.tensors.vp_true.cpu().numpy(), 40)
        )
        print("SUCCESS!")

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
    f = Factory.cli_construct(device=None, src_path=os.path.dirname(__file__))
    f.manufacture_data()


if __name__ == "__main__":
    main()
