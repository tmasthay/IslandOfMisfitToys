# from ...utils import DotDict
# from ..dataset import DataFactory, towed_src, fixed_rec
import os
from os.path import join as pj

import deepwave as dw
import torch

# add_root_package_path(path=os.path.dirname(__file__), pkg="misfit_toys")
from mh.core import DotDict
from scipy.ndimage import gaussian_filter

from misfit_toys.data.dataset import DataFactory, fixed_rec, towed_src


class Factory(DataFactory):
    def _manufacture_data(self):
        if self.installed("vp_true", "vp_strat"):
            return

        self.tensors.vp_true = torch.load(pj(self.src_path, "vmig2A.pt"))
        self.tensors.vp_strat = torch.load(pj(self.src_path, "vstr2A.pt"))


def main():
    f = Factory.cli_construct(device=None, src_path=os.path.dirname(__file__))
    f.manufacture_data()


if __name__ == "__main__":
    main()
