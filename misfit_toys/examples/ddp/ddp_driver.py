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
            prop.get_tensors(), restrict=True, detach=True, device='cpu'
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
            trainer.report.dict(), restrict=True, device='cpu', detach=True
        )

    def final_result(self):
        pass


def main():
    reduce = {
        'loss': torch.stack,
        'obs_data_filt_record': torch.stack,
        'out_record': torch.stack,
        'out_filt_record': torch.stack,
        'vp_record': lambda x: x[0],
        'obs_data': lambda x: x[0],
        'freqs': lambda x: x[0],
        'vp_true': lambda x: x[0],
        'vp_init': lambda x: x[0],
    }

    iomt_example = ExampleIOMT(
        data_save="iomt/data", fig_save="iomt/figs", reduce=reduce, verbose=2
    )
    iomt_example.run()


if __name__ == "__main__":
    main()
