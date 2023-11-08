from misfit_toys.fwi.fwi import FWIPass, FWI
from misfit_toys.utils import get_pydict
from misfit_toys.fwi.modules.models import Param, ParamConstrained
from misfit_toys.fwi.modules.training import TrainingVanilla, TrainingMultiscale
from misfit_toys.utils import taper

import torch
import holoviews as hv


class SourceInversion(FWIPass):
    def _pre_chunk(self, rank, world_size):
        self.prop.obs_data = taper(self.prop.obs_data, 100)
        self.update_tensors(
            self.prop.get_tensors(), restrict=True, detach=True, device=rank
        )
        return self.prop


def main(path):
    hv.extension("matplotlib")

    # path = 'conda/data/openfwi/FlatVel_A'
    path = 'conda/data/marmousi/deepwave_example/shots16/twolayer'
    meta = get_pydict(path, as_class=True)

    extra_forward = {
        'max_vel': 2500,
        'time_pad_frac': 0.2,
        'pml_freq': meta.freq,
    }
    # vp_prmzt = Param.delay_init(requires_grad=True)
    vp_prmzt = ParamConstrained.delay_init(
        requires_grad=False, minv=1000, maxv=2500
    )
    src_amp_y_prmzt = Param.delay_init(requires_grad=True)
    # extra_forward = {}

    prop_kwargs = {
        "path": path,
        "extra_forward_args": extra_forward,
        "vp_prmzt": vp_prmzt,
        "src_amp_y_prmzt": src_amp_y_prmzt,
    }
    reduce = {
        "loss": FWI.mean_reduce,
        "obs_data_filt_record": torch.stack,
        "out_record": torch.stack,
        "out_filt_record": torch.stack,
        "vp_record": FWI.first_elem,
        "obs_data": torch.stack,
        "freqs": FWI.first_elem,
        "vp_true": FWI.first_elem,
        "vp_init": FWI.first_elem,
    }
    verbose = 2

    # loss = torch.nn.HuberLoss(delta=1.0)
    loss = torch.nn.MSELoss()

    freqs = [10.0, 15.0, 20.0, 25.0, 30.0]
    n_epochs = 10
    sub_epochs = n_epochs // len(freqs)
    optimizer = torch.optim.LBFGS
    optimizer_kwargs = dict()

    iomt_example = FWIPass(
        prop_kwargs=prop_kwargs,
        reduce=reduce,
        verbose=verbose,
        training_class=TrainingMultiscale,
        training_kwargs={
            "loss": loss,
            "optimizer": (optimizer, optimizer_kwargs),
            "scheduler": None,
            "freqs": freqs,
            "n_epochs": sub_epochs,
        },
        save_dir="conda/BENCHMARK/multiscale",
    )
    print(iomt_example.prop)
    return iomt_example


if __name__ == "__main__":
    path = "conda/data/marmousi/deepwave_example/shots16"
    ex1 = main(path)
    ex1.run()
