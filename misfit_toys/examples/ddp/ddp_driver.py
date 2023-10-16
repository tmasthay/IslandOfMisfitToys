from misfit_toys.data.dataset import get_data3
from misfit_toys.fwi.modules.models import Param, ParamConstrained
from misfit_toys.fwi.modules.distribution import Distribution, setup, cleanup
from misfit_toys.utils import taper, get_pydict, canonical_reduce
from misfit_toys.fwi.modules.seismic_data import SeismicProp
from misfit_toys.fwi.modules.training import (
    TrainingMultiscale,
    TrainingMultiscaleLegacy,
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
from masthay_helpers.global_helpers import dynamic_expand
import copy


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

    def final_result(self):
        one = {
            "ylabel": "Acoustic Amplitude",
            "loop": {},
            "width": 600,
            "height": 600,
        }
        two = {"loop": {}, "width": 600, "height": 600, "colorbar": True}

        def one_builder(*, data, label_map, base):
            tmp = copy.deepcopy(base)
            tmp["ylim"] = (data.min().item(), data.max().item())
            tmp["loop"]["labels"] = list(label_map.values())
            return tmp

        def two_builder(*, data, label_map, base):
            tmp = copy.deepcopy(base)
            tmp["loop"]["labels"] = list(label_map.values())
            return tmp

        def flatten(data):
            data = [e.reshape(1, -1) for e in data]
            res = torch.stack(data, dim=0)
            second = 2 if res.shape[1] == 1 else 1
            res = res.repeat(1, second, 1)
            return res

        def extend(idx):
            def process(data):
                shape = data[idx].shape
                for i in range(len(data)):
                    if i != idx:
                        data[i] = dynamic_expand(data[i], shape)
                data = torch.stack(data, dim=0)
                return data

            return process

        groups = {
            "Loss": {
                "label_map": {"loss": "Loss"},
                "column_names": ["Frequency", "Epoch"],
                "cols": 1,
                "one": one,
                "two": two,
                "one_builder": one_builder,
                "two_builder": two_builder,
                "data_process": flatten,
            },
            "Obs-Out Filtered": {
                "label_map": {
                    "obs_data_filt_record": "Filtered Observed Data",
                    "out_filt_record": "Filtered Output",
                },
                "column_names": [
                    "Shot",
                    "Receiver",
                    "Time Step",
                    "Frequency",
                    "Epoch",
                ],
                "cols": 2,
                "one": one,
                "two": two,
                "one_builder": one_builder,
                "two_builder": two_builder,
                "data_process": None,
            },
            "Out-Out Filtered": {
                "label_map": {
                    "out_filt_record": "Filtered Output",
                    "out_record": "Output",
                },
                "column_names": [
                    "Shot",
                    "Receiver",
                    "Time Step",
                    "Frequency",
                    "Epoch",
                ],
                "cols": 2,
                "one": one,
                "two": two,
                "one_builder": one_builder,
                "two_builder": two_builder,
                "data_process": None,
            },
            "Velocity": {
                "label_map": {
                    "vp_init": r"$v_0$",
                    "vp_record": r"$v_p$",
                    "vp_true": r"$v_f$",
                },
                "column_names": [
                    "Frequency",
                    "Epoch",
                    "Depth (km)",
                    "Horizontal (km)",
                ],
                "cols": 2,
                "one": one,
                "two": two,
                "one_builder": one_builder,
                "two_builder": two_builder,
                "data_process": extend(1),
            },
        }
        plots = {k: self.plot(**v) for k, v in groups.items()}
        return plots


def main():
    reduce = {
        "loss": Example.mean_reduce,
        "obs_data_filt_record": torch.stack,
        "out_record": torch.stack,
        "out_filt_record": torch.stack,
        "vp_record": Example.first_elem,
        "obs_data": torch.stack,
        "freqs": Example.first_elem,
        "vp_true": Example.first_elem,
        "vp_init": Example.first_elem,
    }

    iomt_example = ExampleIOMT(
        data_save="iomt/data", fig_save="iomt/figs", reduce=reduce, verbose=2
    )
    return iomt_example.run()
