from misfit_toys.data.dataset import get_data3
from misfit_toys.fwi.modules.models import Param, ParamConstrained
from misfit_toys.fwi.modules.distribution import Distribution, setup, cleanup
from misfit_toys.utils import (
    taper,
    get_pydict,
    canonical_reduce,
)
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
        self.tensors.update(prop.get_tensors())
        for k in set(self.tensors.keys()) - set(self.tensor_names):
            del self.tensors[k]

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
        tmp_path = os.path.abspath(os.path.join(self.data_save, "tmp"))
        trainer.train(path=tmp_path)
        self.tensors.update(trainer.report.dict())

        # self.tensors.update(trainer.custom.__dict__)

    def plot_data(self, **kw):
        self.n_epochs = 2
        labels = [
            ("Freq", self.tensors["freqs"]),
            ("Epoch", range(self.n_epochs)),
        ]
        plot_args = dict(
            transpose=True,
            vmin=self.tensors["vp_true"].min(),
            vmax=self.tensors["vp_true"].max(),
            cmap="seismic",
        )
        movie = lambda x: self.plot_inv_record_auto(
            name=x, labels=labels, plot_args=plot_args
        )

        self.plot_loss()
        self.plot_field(field="obs_data", transpose=True, cbar="dynamic")

        movie("vp_record")
        # movie("out_record")
        # movie("out_filt_record")
        # movie("obs_data_filt")

        # note that as of right now "random" does not give you much
        # ... each call will give you a different random subset and thus
        # comparison would be impossible.
        # Maybe you set a seed, but this is then nonrandom.
        idx = ("uniform", 20)
        # self.plot1d(field="out_history", idx=idx)
        # self.plot1d(field="out_filt_history", idx=idx)
        # self.plot1d(field)


def main():
    reduce = canonical_reduce(
        exclude=["src_amp_y"], extra=["vp_record", "vp_true"]
    )

    iomt_example = ExampleIOMT(
        data_save="iomt/data",
        fig_save="iomt/figs",
        reduce=reduce,
        verbose=2,
    )
    iomt_example.run()


if __name__ == "__main__":
    main()
