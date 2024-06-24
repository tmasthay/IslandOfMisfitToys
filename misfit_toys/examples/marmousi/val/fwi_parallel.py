import os
from collections import OrderedDict

import torch
import torch.multiprocessing as mp
from mh.core_legacy import subdict
from scipy.signal import butter
from torch.nn.parallel import DistributedDataParallel as DDP
from torchaudio.functional import biquad

from misfit_toys.data.download_data import download_data
from misfit_toys.fwi.seismic_data import (
    Param,
    ParamConstrained,
    SeismicProp,
    SeismicPropSimple,
    path_builder,
)
from misfit_toys.fwi.training import Training
from misfit_toys.utils import chunk_and_deploy, setup, taper


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


# Main function for training on each rank
def run_rank(rank, world_size, data):
    """
    Runs the Distributed Data Parallel (DDP) training on a specific rank.

    Args:
        rank (int): The rank of the current process.
        world_size (int): The total number of processes.

    Returns:
        None
    """

    print(f"Running DDP on rank {rank} / {world_size}.", flush=True)
    setup(rank, world_size, port=12355)

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
    prop = SeismicPropSimple(
        **prop_data,
        forward_kw=dict(
            max_vel=2500, pml_freq=data["meta"].freq, time_pad_frac=0.2
        ),
    ).to(rank)
    prop = DDP(prop, device_ids=[rank])

    # Setup optimiser to perform inversion
    loss_fn = torch.nn.MSELoss()

    # Run optimisation/inversion
    n_epochs = 2

    loss_record = []
    v_record = []
    out_record = []
    out_filt_record = []

    freqs = [10, 15, 20, 25, 30]
    # n_freqs = len(freqs)

    def get_epoch(i, j):
        return j + i * n_epochs

    obs_data = data['obs_data']
    for i, cutoff_freq in enumerate(freqs):
        print(i, flush=True)
        sos = butter(6, cutoff_freq, fs=1 / data["meta"].dt, output='sos')
        sos = [torch.tensor(sosi).to(obs_data.dtype).to(rank) for sosi in sos]

        def filt(x):
            return biquad(biquad(biquad(x, *sos[0]), *sos[1]), *sos[2])

        observed_data_filt = filt(obs_data)
        optimiser = torch.optim.LBFGS(prop.parameters())
        for epoch in range(n_epochs):
            num_calls = 0

            def closure():
                nonlocal num_calls
                num_calls += 1
                optimiser.zero_grad()
                out = prop(None)
                out_filt = filt(taper(out[-1]))
                loss = 1e6 * loss_fn(out_filt, observed_data_filt)
                loss.backward()
                if num_calls == 1:
                    loss_record.append(loss.detach().cpu())
                    v_record.append(prop.module.vp().detach().cpu())
                    out_record.append(out[-1].detach().cpu())
                    out_filt_record.append(out_filt.detach().cpu())
                    print(
                        f'Epoch={get_epoch(i, epoch)}, Loss={loss.item()},'
                        f' rank={rank}',
                        flush=True,
                    )
                return loss

            optimiser.step(closure)
            torch.cuda.empty_cache()
    print(f'Exiting {rank=}')


# Main function for spawning ranks
def run(world_size):
    """
    Runs the FWI parallel process.

    Args:
        world_size (int): The number of processes to spawn.

    Returns:
        None
    """
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
    mp.spawn(run_rank, args=(world_size, data), nprocs=world_size, join=True)


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
