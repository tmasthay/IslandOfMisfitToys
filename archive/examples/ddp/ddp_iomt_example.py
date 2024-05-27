import os
from time import time

import deepwave
import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from deepwave import scalar
from example import Example
from scipy.ndimage import gaussian_filter
from scipy.signal import butter
from torch.nn.parallel import DistributedDataParallel as DDP
from torchaudio.functional import biquad

from misfit_toys.data.dataset import get_data3
from misfit_toys.fwi.modules.distribution import Distribution, cleanup, setup
from misfit_toys.fwi.modules.models import Param, ParamConstrained
from misfit_toys.fwi.modules.seismic_data import SeismicProp
from misfit_toys.utils import get_pydict, print_tensor, taper


class ExampleRudolph(Example):
    def _generate_data(self, rank, world_size):
        prop = SeismicProp(
            path="conda/data/marmousi/deepwave_example/shots16",
            extra_forward_args={"time_pad_frac": 0.2},
            vp_prmzt=ParamConstrained.delay_init(minv=1000, maxv=2500),
        )
        print(prop)

    def plot_data(self, **kw):
        pass


example = ExampleRudolph(
    data_save="iomt_output/data",
    fig_save="iomt_output/figs",
    tensor_names=["vp_true", "vp_init", "obs_data"],
    verbose=1,
)
example.run()
