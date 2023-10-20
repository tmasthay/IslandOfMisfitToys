from misfit_toys.data.dataset import get_data3
from misfit_toys.fwi.modules.models import Param, ParamConstrained
from misfit_toys.fwi.modules.distribution import Distribution, setup, cleanup
from misfit_toys.utils import print_tensor, taper, get_pydict
from misfit_toys.fwi.modules.seismic_data import SeismicProp

import os
import torch
from torchaudio.functional import biquad
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from scipy.ndimage import gaussian_filter
from scipy.signal import butter
import matplotlib.pyplot as plt
import deepwave
from deepwave import scalar
from time import time

from example import Example


class ExampleRudolph(Example):
    def _generate_data(self, rank, world_size):
        prop = SeismicProp(
            path="conda/data/marmousi/deepwave_example/shots16",
            extra_forward_args={"time_pad_frac": 0.2},
            vp_prmzt=ParamConstrained.delay_init(minv=1000, maxv=2500),
        )

    def plot_data(self, **kw):
        pass


example = ExampleRudolph(
    data_save="iomt_output/data",
    fig_save="iomt_output/figs",
    tensor_names=["vp_true", "vp_init", "obs_data"],
    verbose=1,
)
example.run()
