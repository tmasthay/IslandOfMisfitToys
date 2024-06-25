import os
from collections import OrderedDict

import torch
import torch.multiprocessing as mp
from mh.core_legacy import subdict
from scipy.signal import butter
from torch.nn.parallel import DistributedDataParallel as DDP

from misfit_toys.data.download_data import download_data
from misfit_toys.fwi.seismic_data import (
    Param,
    ParamConstrained,
    SeismicPropSimple,
    path_builder,
    Model
)
from misfit_toys.fwi.training import Training
from misfit_toys.utils import chunk_and_deploy, filt, setup, taper
from deepwave import scalar


def training_stages():
    """
    Define the training stages for the training class.

    Returns:
        OrderedDict: A dictionary containing the training stages.
            Each stage is represented by a key-value pair, where the key is the stage name
            and the value is a dictionary containing the stage data, preprocess function, and postprocess function.
    """

    def freq_preprocess(training, freq):
        sos = butter(6, freq, fs=1 / training.prop.module.dt, output="sos")
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
# def _step(self):
#     """
#     Performs a single step of the forward-backward optimization process.

#     Returns:
#         torch.Tensor: The loss value after the backward pass.
#     """
#     self.out = self.prop(None)
#     self.out_filt = filt(taper(self.out[-1]), self.sos)
#     self.loss = 1e6 * self.loss_fn(self.out_filt, self.obs_data_filt)
#     self.loss.backward()
#     return self.loss


# Syntactic sugar for converting from device to cpu
def d2cpu(x):
    """
    Moves a tensor `x` from GPU to CPU and detaches it from the computation graph.

    Args:
        x (torch.Tensor): The input tensor to be moved from GPU to CPU.

    Returns:
        torch.Tensor: The tensor `x` moved to CPU and detached from the computation graph.
    """
    return x.detach().cpu()

class Model(torch.nn.Module):
    def __init__(self, initial, min_vel, max_vel):
        super().__init__()
        self.min_vel = min_vel
        self.max_vel = max_vel
        self.model = torch.nn.Parameter(
            torch.logit((initial - min_vel) / (max_vel - min_vel))
        )

    def forward(self):
        return (
            torch.sigmoid(self.model) * (self.max_vel - self.min_vel)
            + self.min_vel
        )


class Prop(torch.nn.Module):
    def __init__(self, *, model, dx, dt, freq):
        super().__init__()
        self.vp = model
        self.dx = dx
        self.dt = dt
        self.freq = freq

    def forward(self, *, src_amp_y, rec_loc_y, src_loc_y):
        v = self.vp()
        return scalar(
            v,
            self.dx,
            self.dt,
            source_amplitudes=src_amp_y,
            source_locations=src_loc_y,
            receiver_locations=rec_loc_y,
            max_vel=2500,
            pml_freq=self.freq,
            time_pad_frac=0.2,
        )
    
# Main function for training on each rank
def run_rank(rank, world_size):
    """
    Runs the Distributed Data Parallel (DDP) training on a specific rank.

    Args:
        rank (int): The rank of the current process.
        world_size (int): The total number of processes.

    Returns:
        None
    """

    print(f"Running DDP on rank {rank} / {world_size}.")
    setup(rank, world_size)

    data_path = os.path.join(
        os.environ['CONDA_PREFIX'], 'data/marmousi/deepwave_example/shots16'
    )
    if not os.path.exists(data_path):
        print("Downloading data...")
        download_data(storage="conda/data", inclusions={"marmousi"})
        print("Data downloaded.")
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
    model=Model(data['vp'](), 1000, 2500)
    dx=prop_data['meta']['dx']
    dt = prop_data['meta']['dt']
    pml_freq = prop_data['meta']['freq']
    src_amp_y = prop_data['src_amp_y']().to(rank)
    src_loc_y = prop_data['src_loc_y'].to(rank)
    rec_loc_y = prop_data['rec_loc_y'].to(rank)
    forward_kw = dict(max_vel=2500, pml_freq=pml_freq, time_pad_frac=0.2)

    prop = Prop(model=model, 
                dx=dx, dt=dt, freq=pml_freq).to(rank)
    
    prop = DDP(prop, device_ids=[rank])


    def _step(self):
        """
        Performs a single step of the forward-backward optimization process.

        Returns:
            torch.Tensor: The loss value after the backward pass.
        """
        self.out = self.prop(src_amp_y=src_amp_y, rec_loc_y=rec_loc_y, src_loc_y=src_loc_y)
        self.out_filt = filt(taper(self.out[-1]), self.sos)
        self.loss = 1e6 * self.loss_fn(self.out_filt, self.obs_data_filt)
        self.loss.backward()
        return self.loss

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
    """
    Runs the FWI parallel process.

    Args:
        world_size (int): The number of processes to spawn.

    Returns:
        None
    """
    mp.spawn(run_rank, args=(world_size,), nprocs=world_size, join=True)


# Main function for running the script
def main():
    """
    Main function to run the FWI parallel program.

    Args:
        None

    Returns:
        None
    """
    n_gpus = torch.cuda.device_count()
    run(n_gpus)


# Run the script from command line
if __name__ == "__main__":
    main()
