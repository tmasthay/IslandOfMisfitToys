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
        if self.installed('vp_true', 'src_loc_y', 'rec_loc_y', 'obs_data'):
            return

        self.tensors = self.get_parent_tensors()
        d = DotDict(self.metadata)

        v_slice, src_slice, rec_slice = DataFactory.get_slices(d)

        print('Slicing shots16 tensors...', end='', flush=True)
        self.slice_subset_tensors(
            *src_slice,
            keys=['src_loc_y', 'src_amp_y', 'src_loc_x', 'src_amp_x'],
        )
        self.slice_subset_tensors(
            *rec_slice, keys=['rec_loc_y', 'rec_loc_x', 'obs_data']
        )

        print('SUCCESS!')


def main():
    f = Factory.cli_construct(
        device='cuda:0', src_path=os.path.dirname(__file__)
    )
    f.manufacture_data()


if __name__ == "__main__":
    main()
