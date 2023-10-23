from misfit_toys.data.dataset import get_data3
from misfit_toys.fwi.modules.models import Param, ParamConstrained
from misfit_toys.fwi.modules.distribution import Distribution, setup, cleanup
from misfit_toys.utils import taper, get_pydict, canonical_reduce
from misfit_toys.fwi.modules.seismic_data import SeismicProp
from misfit_toys.fwi.modules.training import (
    TrainingMultiscale,
    TrainingMultiscaleLegacy,
    TrainingVanilla,
)
from misfit_toys.utils import idt_print

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
from misfit_toys.examples.example import Example
from masthay_helpers.jupyter import iplot_workhorse
from masthay_helpers.global_helpers import dynamic_expand, prettify_dict
import copy
from misfit_toys.fwi.custom_losses import W1, Renorm, L2

from rich.traceback import install

install(show_locals=True)


class ExampleIOMT(Example):
    def _pre_chunk(self, rank, world_size):
        self.prop.obs_data = taper(self.prop.obs_data, 100)
        self.update_tensors(
            self.prop.get_tensors(), restrict=True, detach=True, device=rank
        )
        return self.prop


def main():
    path = "conda/data/marmousi/deepwave_example/shots16"
    meta = get_pydict(path, as_class=True)
    iomt_example = ExampleIOMT(
        prop_kwargs={
            "path": path,
            "extra_forward_args": {
                "max_vel": 2500,
                "time_pad_frac": 0.2,
                "pml_freq": meta.freq,
            },
            "vp_prmzt": ParamConstrained.delay_init(
                requires_grad=True, minv=1000, maxv=2500
            ),
            "src_amp_y_prmzt": Param.delay_init(requires_grad=False),
        },
        training_class=TrainingMultiscale,
        training_kwargs={
            "loss": torch.nn.MSELoss(),
            "optimizer": (torch.optim.LBFGS, dict()),
            "scheduler": None,
            "verbose": 1,
            "freqs": [10.0, 15.0, 20.0, 25.0, 30.0],
            "n_epochs": 2,
        },
        reduce={
            "loss": ExampleIOMT.mean_reduce,
            "obs_data_filt_record": torch.stack,
            "out_record": torch.stack,
            "out_filt_record": torch.stack,
            "vp_record": ExampleIOMT.first_elem,
            "obs_data": torch.stack,
            "freqs": ExampleIOMT.first_elem,
            "vp_true": ExampleIOMT.first_elem,
            "vp_init": ExampleIOMT.first_elem,
        },
        save_dir="/home/tyler",
        verbose=2,
    )
    return iomt_example.run()

    # w1_example = ExampleGen(
    #     path="conda/data/marmousi/deepwave_example/shots16",
    #     data_save="w2/data",
    #     fig_save="w2/figs",
    #     loss=W1(renorm_func=Renorm.choose("exp")),
    #     optimizer=(torch.optim.LBFGS, dict()),
    #     scheduler=None,
    #     reduce={
    #         "loss": Example.mean_reduce,
    #         "out_record": torch.stack,
    #         "vp_record": Example.first_elem,
    #         "obs_data": torch.stack,
    #         "vp_true": Example.first_elem,
    #         "vp_init": Example.first_elem,
    #     },
    #     verbose=2,
    # )

    # iomt_output = iomt_example.run()
    # w1_output = w1_example.run()
    # return iomt_output, w1_output


if __name__ == "__main__":
    main()
