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


class ExampleIOMT(Example):
    def _generate_data(self, rank, world_size):
        path = "conda/data/marmousi/deepwave_example/shots16"
        meta = get_pydict(path, as_class=True)
        chunk_size = meta.n_shots // world_size
        amp_idx = torch.arange(
            rank * chunk_size, (rank + 1) * chunk_size, dtype=torch.long
        )
        prop = SeismicProp(
            path="conda/data/marmousi/deepwave_example/shots16",
            extra_forward_args={
                "max_vel": 2500,
                "time_pad_frac": 0.2,
                "pml_freq": meta.freq,
                "amp_idx": amp_idx,
            },
            vp_prmzt=ParamConstrained.delay_init(
                requires_grad=True, minv=1000, maxv=2500
            ),
            src_amp_y_prmzt=Param.delay_init(requires_grad=False),
        )
        prop.obs_data = taper(prop.obs_data, 100)
        self.update_tensors(
            prop.get_tensors(), restrict=True, detach=True, device="cpu"
        )

        prop = prop.chunk(rank, world_size)
        prop = prop.to(rank)
        dist_prop = DDP(prop, device_ids=[rank])
        trainer = TrainingMultiscale(
            dist_prop=dist_prop,
            rank=rank,
            world_size=world_size,
            optimizer=(torch.optim.LBFGS, dict()),
            loss=torch.nn.MSELoss(),
            scheduler=None,
            verbose=1,
            freqs=[10.0, 15.0, 20, 25, 30],
            n_epochs=2,
            reduce=self.reduce,
        )
        # trainer = TrainingMultiscaleLegacy(
        #     dist_prop=dist_prop, rank=rank, world_size=world_size
        # )

        # TODO: Your I/O seems inefficient.
        #    This is low-priority since we have a compute bound task, but
        #    monitor this for large problem size as it may become
        #    prohibitive for no good reason.
        tmp_path = os.path.abspath(os.path.join(self.data_save, "tmp"))
        trainer.train(path=tmp_path)
        self.update_tensors(
            trainer.report.dict(), restrict=True, device="cpu", detach=True
        )


class ExampleGen(Example):
    def __init__(
        self,
        *,
        loss,
        optimizer,
        scheduler=None,
        data_save,
        fig_save,
        reduce,
        verbose
    ):
        super().__init__(
            loss=loss,
            optimizer=optimizer,
            scheduler=scheduler,
            data_save=data_save,
            fig_save=fig_save,
            reduce=reduce,
            verbose=verbose,
        )

    def _generate_data(self, rank, world_size):
        path = "conda/data/marmousi/deepwave_example/shots16"
        meta = get_pydict(path, as_class=True)
        chunk_size = meta.n_shots // world_size
        amp_idx = torch.arange(
            rank * chunk_size, (rank + 1) * chunk_size, dtype=torch.long
        )
        prop = SeismicProp(
            path="conda/data/marmousi/deepwave_example/shots16",
            extra_forward_args={
                "max_vel": 2500,
                "time_pad_frac": 0.2,
                "pml_freq": meta.freq,
                "amp_idx": amp_idx,
            },
            vp_prmzt=ParamConstrained.delay_init(
                requires_grad=True, minv=1000, maxv=2500
            ),
            src_amp_y_prmzt=Param.delay_init(requires_grad=False),
        )
        prop.obs_data = taper(prop.obs_data, 100)
        self.update_tensors(
            prop.get_tensors(), restrict=True, detach=True, device="cpu"
        )

        prop = prop.chunk(rank, world_size)
        prop = prop.to(rank)
        dist_prop = DDP(prop, device_ids=[rank])
        trainer = TrainingVanilla(
            dist_prop=dist_prop,
            rank=rank,
            world_size=world_size,
            optimizer=self.optimizer,
            loss=self.loss,
            scheduler=self.scheduler,
            verbose=1,
            n_epochs=5,
            reduce=self.reduce,
        )

        tmp_path = os.path.abspath(os.path.join(self.data_save, "tmp"))
        trainer.train(path=tmp_path)
        self.update_tensors(
            trainer.report.dict(), restrict=True, device="cpu", detach=True
        )

    def _final_dict(self):
        u = self.base_final_dict()
        del u["Out-Out Filtered"], u["Obs-Out Filtered"]

        u["Velocity"]["column_names"] = [
            "Epoch",
            "Depth (km)",
            "Horizontal (km)",
        ]
        u["Obs-Out"]["column_names"] = [
            "Shot",
            "Receiver",
            "Time Step",
            "Epoch",
        ]
        input(prettify_dict(u))
        return u


def main():
    iomt_example = ExampleIOMT(
        data_save="iomt/data",
        fig_save="iomt/figs",
        reduce={
            "loss": Example.mean_reduce,
            "obs_data_filt_record": torch.stack,
            "out_record": torch.stack,
            "out_filt_record": torch.stack,
            "vp_record": Example.first_elem,
            "obs_data": torch.stack,
            "freqs": Example.first_elem,
            "vp_true": Example.first_elem,
            "vp_init": Example.first_elem,
        },
        verbose=2,
    )

    w1_example = ExampleGen(
        data_save="w2/data",
        fig_save="w2/figs",
        loss=W1(renorm_func=Renorm.choose("exp")),
        optimizer=(torch.optim.LBFGS, dict()),
        scheduler=None,
        reduce={
            "loss": Example.mean_reduce,
            "out_record": torch.stack,
            "vp_record": Example.first_elem,
            "obs_data": torch.stack,
            "vp_true": Example.first_elem,
            "vp_init": Example.first_elem,
        },
        verbose=2,
    )

    iomt_output = iomt_example.run()
    w1_output = w1_example.run()
    return iomt_output, w1_output
