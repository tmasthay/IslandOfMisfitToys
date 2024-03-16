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
            "rho_true",
            "vs_true",
            "obs_data",
            "src_loc_y",
            "rec_loc_y",
            "src_amp_y",
        ):
            return

        self.tensors = self.get_parent_tensors()
        d = DotDict(self.metadata)

        v_slice, src_slice, rec_slice = DataFactory.get_slices(d)
        self.slice_subset_tensors(
            *v_slice, keys=["vp_true", "vs_true", "rho_true"]
        )
        self.slice_subset_tensors(
            *src_slice,
            keys=["src_loc_y", "src_amp_y", "src_loc_x", "src_amp_x"],
        )
        self.slice_subset_tensors(*rec_slice, keys=["rec_loc_y", "rec_loc_x"])

        print("Building obs_data in marmousi2/medium...", end="", flush=True)
        self.tensors.obs_data = dw.elastic(
            *get_lame(
                self.tensors.vp_true,
                self.tensors.vs_true,
                self.tensors.rho_true,
            ),
            d.dx,
            d.dt,
            source_amplitudes_y=self.tensors.src_amp_y,
            source_locations_y=self.tensors.src_loc_y,
            receiver_locations_y=self.tensors.rec_loc_y,
            pml_freq=d.freq,
            accuracy=d.accuracy,
        )[-2].to("cpu")
        print("SUCCESS", flush=True)


def main():
    f = Factory.cli_construct(
        device="cuda:0", src_path=os.path.dirname(__file__)
    )
    f.manufacture_data()


if __name__ == "__main__":
    main()
