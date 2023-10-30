from misfit_toys.data.dataset import get_data3
from misfit_toys.fwi.modules.models import Param, ParamConstrained
from misfit_toys.fwi.modules.distribution import Distribution, setup, cleanup
from misfit_toys.utils import taper, get_pydict, canonical_reduce
from misfit_toys.fwi.modules.seismic_data import SeismicProp
from misfit_toys.fwi.modules.training import TrainingMultiscale, TrainingVanilla
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
from masthay_helpers.global_helpers import (
    dynamic_expand,
    prettify_dict,
    extend_dict,
)
import copy
from misfit_toys.fwi.custom_losses import W1, Renorm, L2, HuberLoss
from returns.curry import partial
import holoviews as hv

# from rich.traceback import install

# install(show_locals=True)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class ExampleIOMT(Example):
    def _pre_chunk(self, rank, world_size):
        self.prop.obs_data = taper(self.prop.obs_data, 100)
        self.update_tensors(
            self.prop.get_tensors(), restrict=True, detach=True, device=rank
        )
        return self.prop


class Example2(ExampleIOMT):
    def _final_dict(self):
        u = extend_dict(
            self.base_final_dict(), sub=["Obs-Out Filtered", "Out-Out Filtered"]
        )
        for k, v in u.items():
            if "column_names" in v.keys():
                u[k]["column_names"].remove("Frequency")
            if len(u[k]["column_names"]) == 1:
                u[k]["column_names"].append(u[k]["column_names"][0])
        return u


def main():
    hv.extension("matplotlib")
    path = "conda/data/marmousi/deepwave_example/shots16"
    # path = 'conda/data/openfwi/FlatVel_A'
    meta = get_pydict(path, as_class=True)

    extra_forward = {
        'max_vel': 2500,
        'time_pad_frac': 0.2,
        'pml_freq': meta.freq,
    }
    vp_prmzt = Param.delay_init(requires_grad=True)
    # vp_prmzt = ParamConstrained.delay_init(
    #     requires_grad=True, minv=1000, maxv=2500
    # )
    src_amp_y_prmzt = Param.delay_init(requires_grad=False)
    extra_forward = {}

    prop_kwargs = {
        "path": path,
        "extra_forward_args": extra_forward,
        "vp_prmzt": vp_prmzt,
        "src_amp_y_prmzt": src_amp_y_prmzt,
    }
    reduce = {
        "loss": ExampleIOMT.mean_reduce,
        "obs_data_filt_record": torch.stack,
        "out_record": torch.stack,
        "out_filt_record": torch.stack,
        "vp_record": ExampleIOMT.first_elem,
        "obs_data": torch.stack,
        "freqs": ExampleIOMT.first_elem,
        "vp_true": ExampleIOMT.first_elem,
        "vp_init": ExampleIOMT.first_elem,
    }
    verbose = 2

    iomt_example = ExampleIOMT(
        prop_kwargs=prop_kwargs,
        reduce=reduce,
        verbose=verbose,
        training_class=TrainingMultiscale,
        training_kwargs={
            "loss": HuberLoss(delta=0.1),
            "optimizer": (torch.optim.LBFGS, dict()),
            "scheduler": None,
            "freqs": [10.0, 15.0, 20.0, 25.0, 30.0],
            "n_epochs": 2,
        },
        save_dir="conda/BENCHMARK/multiscale",
    )
    example2 = Example2(
        prop_kwargs=prop_kwargs,
        reduce=extend_dict(
            reduce, sub=(["freqs", "obs_data_filt_record", "out_filt_record"])
        ),
        verbose=verbose,
        training_class=TrainingVanilla,
        training_kwargs={
            "loss": torch.nn.MSELoss(),
            "optimizer": (torch.optim.LBFGS, dict()),
            "scheduler": None,
            "n_epochs": 10,
        },
        save_dir="conda/BENCHMARK/vanilla",
    )
    first = iomt_example.run()
    second = example2.run()

    return first, second

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
