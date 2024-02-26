import os
from collections import OrderedDict

import torch
import torch.multiprocessing as mp
from mh.core_legacy import subdict
from scipy.signal import butter
from torch.nn.parallel import DistributedDataParallel as DDP

from misfit_toys.fwi.seismic_data import (
    Param,
    ParamConstrained,
    SeismicProp,
    path_builder,
)
from misfit_toys.fwi.training import Training
from misfit_toys.utils import chunk_and_deploy, filt, setup, taper


def training_stages():
    # define training stages for the training class
    def freq_preprocess(training, freq):
        sos = butter(6, freq, fs=1 / training.prop.module.meta.dt, output="sos")
        sos = [torch.tensor(sosi).to(training.obs_data.dtype) for sosi in sos]

        training.sos = sos

        training.obs_data_filt = filt(training.obs_data, sos)

        training.reset_optimizer()

    def freq_postprocess(training, freq):
        pass

    def epoch_preprocess(training, epoch):
        pass

    def epoch_postprocess(training, epoch):
        pass

    return OrderedDict(
        [
            (
                "freqs",
                {
                    "data": [10, 15, 20, 25, 30],
                    "preprocess": freq_preprocess,
                    "postprocess": freq_postprocess,
                },
            ),
            (
                "epochs",
                {
                    "data": [0, 1],
                    "preprocess": epoch_preprocess,
                    "postprocess": epoch_postprocess,
                },
            ),
        ]
    )


# Define _step for the training class
def _step(self):
    self.out = self.prop(None)
    self.out_filt = filt(taper(self.out[-1]), self.sos)
    self.loss = 1e6 * self.loss_fn(self.out_filt, self.obs_data_filt)
    self.loss.backward()
    return self.loss


# Syntactic sugar for converting from device to cpu
def d2cpu(x):
    return x.detach().cpu()


# Main function for training on each rank
def run_rank(rank, world_size):
    print(f"Running DDP on rank {rank} / {world_size}.")
    setup(rank, world_size)

    # Build data for marmousi model
    data = path_builder(
        "conda/data/marmousi/deepwave_example/shots16",
        remap={"vp_init": "vp"},
        vp_init=ParamConstrained.delay_init(
            minv=1000, maxv=2500, requires_grad=True
        ),
        src_amp_y=Param.delay_init(requires_grad=False),
        obs_data=None,
        src_loc_y=None,
        rec_loc_y=None,
    )

    # preprocess data like Alan and then deploy slices onto GPUs
    data["obs_data"] = taper(data["obs_data"])
    data = chunk_and_deploy(
        rank,
        world_size,
        data=data,
        chunk_keys={
            "tensors": ["obs_data", "src_loc_y", "rec_loc_y"],
            "params": ["src_amp_y"],
        },
    )

    # Build seismic propagation module and wrap in DDP
    prop_data = subdict(data, exc=["obs_data"])
    prop = SeismicProp(
        **prop_data, max_vel=2500, pml_freq=data["meta"].freq, time_pad_frac=0.2
    ).to(rank)
    prop = DDP(prop, device_ids=[rank])

    # Define the training object
    train = Training(
        rank=rank,
        world_size=world_size,
        prop=prop,
        obs_data=data["obs_data"],
        loss_fn=torch.nn.MSELoss(),
        optimizer=[torch.optim.LBFGS, {}],
        verbose=2,
        report_spec={
            'path': os.path.join(os.path.dirname(__file__), 'out', 'parallel'),
            'loss': {
                'update': lambda x: d2cpu(x.loss),
                'reduce': lambda x: torch.mean(torch.stack(x), dim=0),
                'presave': torch.tensor,
            },
            'vp': {
                'update': lambda x: d2cpu(x.prop.module.vp()),
                'reduce': lambda x: x[0],
                'presave': torch.stack,
            },
            'out': {
                'update': lambda x: d2cpu(x.out[-1]),
                'reduce': lambda x: torch.cat(x, dim=1),
                'presave': torch.stack,
            },
            'out_filt': {
                'update': lambda x: d2cpu(x.out_filt),
                'reduce': lambda x: torch.cat(x, dim=1),
                'presave': torch.stack,
            },
        },
        _step=_step,
        _build_training_stages=training_stages,
    )
    train.train()


# Main function for spawning ranks
def run(world_size):
    mp.spawn(run_rank, args=(world_size,), nprocs=world_size, join=True)


# Main function for running the script
def main():
    n_gpus = torch.cuda.device_count()
    run(n_gpus)


# Run the script from command line
if __name__ == "__main__":
    main()
