from misfit_toys.data.dataset import (
    field_getter,
    field_saver,
    get_data3,
    fetch_warn,
)

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
from warnings import warn
from itertools import product as prod
import pickle

# from ..example import Example, define_names
from misfit_toys.examples.example import Example, define_names


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
    def __init__(self, model, dx, dt, freq):
        super().__init__()
        self.model = model
        self.dx = dx
        self.dt = dt
        self.freq = freq

    def forward(self, source_amplitudes, source_locations, receiver_locations):
        v = self.model()
        return scalar(
            v,
            self.dx,
            self.dt,
            source_amplitudes=source_amplitudes,
            source_locations=source_locations,
            receiver_locations=receiver_locations,
            max_vel=2500,
            pml_freq=self.freq,
            time_pad_frac=0.2,
        )


class MultiscaleExample(Example):
    def _generate_data(self, rank, world_size):
        print(f"self.address at top of _generate_data={id(self)}", flush=True)
        print(
            (
                "self.tensors.keys() at top of "
                f" _generate_data={self.tensors.keys()} with rank={rank}"
            ),
            flush=True,
        )
        ny = 2301
        nx = 751
        dx = 4.0

        has_vp_bin = True
        try:
            v_true = torch.from_file("marmousi_vp.bin", size=ny * nx).reshape(
                ny, nx
            )
        except:
            self.print("v_true from_file upload failed!")
            v_true = None

        self.print(f"self.tensors == {self.tensors}")
        if v_true is None:
            fetch_warn()
            has_vp_bin = False
            get = field_getter("conda/data/marmousi")
            v_true = get("vp_true")

        # Select portion of model for inversion
        ny = 600
        nx = 250
        v_true = v_true[:ny, :nx]
        self.tensors["vp_true"] = v_true

        # Smooth to use as starting model
        v_init = torch.tensor(1 / gaussian_filter(1 / v_true.numpy(), 40))

        n_shots = 115

        n_sources_per_shot = 1
        d_source = 20  # 20 * 4m = 80m
        first_source = 10  # 10 * 4m = 40m
        source_depth = 2  # 2 * 4m = 8m

        n_receivers_per_shot = 384
        d_receiver = 6  # 6 * 4m = 24m
        first_receiver = 0  # 0 * 4m = 0m
        receiver_depth = 2  # 2 * 4m = 8m

        freq = 25
        nt = 750
        dt = 0.004
        peak_time = 1.5 / freq

        try:
            observed_data = torch.from_file(
                "marmousi_data.bin", size=n_shots * n_receivers_per_shot * nt
            ).reshape(n_shots, n_receivers_per_shot, nt)
        except:
            if has_vp_bin:
                raise ValueError(
                    "See code...has should be impossible "
                    "when trying to fetch observed_data"
                )
            fetch_warn()
            observed_data = get("obs_data")

        def taper(x):
            # Taper the ends of traces
            return deepwave.common.cosine_taper_end(x, 100)

        # Select portion of data for inversion
        n_shots = 16
        n_receivers_per_shot = 100
        nt = 300
        observed_data = taper(
            observed_data[:n_shots, :n_receivers_per_shot, :nt]
        )

        # source_locations
        source_locations = torch.zeros(
            n_shots, n_sources_per_shot, 2, dtype=torch.long
        )
        source_locations[..., 1] = source_depth
        source_locations[:, 0, 0] = (
            torch.arange(n_shots) * d_source + first_source
        )

        # receiver_locations
        receiver_locations = torch.zeros(
            n_shots, n_receivers_per_shot, 2, dtype=torch.long
        )
        receiver_locations[..., 1] = receiver_depth
        receiver_locations[:, :, 0] = (
            torch.arange(n_receivers_per_shot) * d_receiver + first_receiver
        ).repeat(n_shots, 1)

        # source_amplitudes
        source_amplitudes = (
            deepwave.wavelets.ricker(freq, nt, dt, peak_time)
        ).repeat(n_shots, n_sources_per_shot, 1)

        self.tensors["src_loc_y"] = source_locations
        self.tensors["rec_loc_y"] = receiver_locations
        self.tensors["src_amp_y"] = source_amplitudes
        self.tensors["obs_data"] = observed_data

        torch.save(
            self.tensors["obs_data"].detach().cpu(), "/home/tyler/obs_data.pt"
        )

        print(f"Rank={rank}, id={id(observed_data)}", flush=True)
        observed_data = torch.chunk(observed_data, world_size)[rank].to(rank)
        source_amplitudes = torch.chunk(source_amplitudes, world_size)[rank].to(
            rank
        )
        source_locations = torch.chunk(source_locations, world_size)[rank].to(
            rank
        )
        receiver_locations = torch.chunk(receiver_locations, world_size)[
            rank
        ].to(rank)

        model = Model(v_init, 1000, 2500)
        self.tensors["vp_init"] = model().detach().cpu()
        prop = Prop(model, dx, dt, freq).to(rank)
        prop = DDP(prop, device_ids=[rank])

        # Setup optimiser to perform inversion
        loss_fn = torch.nn.MSELoss()

        # Run optimisation/inversion
        n_epochs = 2
        self.n_epochs = n_epochs

        self.tensors["freqs"] = torch.Tensor([10, 15, 20, 25, 30])
        self.tensors["vp_record"] = torch.zeros(
            self.tensors["freqs"].shape[0], n_epochs, *v_true.shape
        )

        self.tensors["loss"] = (
            torch.zeros(
                world_size, self.tensors["freqs"].shape[0], self.n_epochs
            )
            .detach()
            .cpu()
        )

        self.print(f"TRAIN BEGIN, Rank={rank}")
        freqs = self.tensors["freqs"]
        loss_local = torch.zeros(freqs.shape[0], n_epochs).to(rank)
        if rank == 0:
            gather_loss = [
                torch.zeros_like(loss_local) for _ in range(world_size)
            ]
        else:
            gather_loss = None

        for idx, cutoff_freq in enumerate(freqs):
            sos = butter(6, cutoff_freq, fs=1 / dt, output="sos")
            sos = [
                torch.tensor(sosi).to(observed_data.dtype).to(rank)
                for sosi in sos
            ]

            def filt(x):
                return biquad(biquad(biquad(x, *sos[0]), *sos[1]), *sos[2])

            observed_data_filt = filt(observed_data)
            optimiser = torch.optim.LBFGS(prop.parameters())

            for epoch in range(n_epochs):
                epoch_loss = 0.0
                closure_calls = 0

                def closure():
                    nonlocal closure_calls, epoch_loss
                    closure_calls += 1
                    optimiser.zero_grad()
                    out = prop(
                        source_amplitudes, source_locations, receiver_locations
                    )
                    out_filt = filt(taper(out[-1]))
                    loss = 1e6 * loss_fn(out_filt, observed_data_filt)
                    if closure_calls == 1:
                        loss_local[idx, epoch] = loss.item()
                        epoch_loss = loss_local[idx, epoch]
                    loss.backward()
                    return loss

                optimiser.step(closure)
                self.tensors["vp_record"][idx, epoch] = model().detach().cpu()
                print(
                    (
                        f"Loss={epoch_loss:.16f}, "
                        f"Freq={cutoff_freq}, "
                        f"Epoch={epoch}, "
                        f"Rank={rank}"
                    ),
                    flush=True,
                )
        self.print(f"TRAIN END, Rank={rank}")
        self.print(loss_local)
        torch.distributed.gather(
            tensor=loss_local, gather_list=gather_loss, dst=0
        )
        if rank == 0:
            self.print(f"GATHER BEGIN, Rank={rank}")
            self.tensors["loss"] = torch.stack(gather_loss).to("cpu")
            self.print(
                f'Gathered data: {self.tensors["loss"]} of shape'
                f' {self.tensors["loss"].shape}'
            )

        print(
            f"self.address at bottom of _generate_data={id(self)}", flush=True
        )
        print(
            (
                "self.tensors.keys at bottom of"
                f" _generate_data={self.tensors.keys()} with rank={rank}"
            ),
            flush=True,
        )

    def plot_data(self, **kw):
        print(f"self.address top of plot_data={id(self)}", flush=True)
        print(
            f"self.tensors.keys() at top of plot_data={self.tensors.keys()}",
            flush=True,
        )
        self.n_epochs = 2
        self.plot_inv_record_auto(
            name="vp",
            labels=[
                ("Freq", self.tensors["freqs"]),
                ("Epoch", range(self.n_epochs)),
            ],
            plot_args=dict(
                transpose=True,
                vmin=self.tensors["vp_true"].min(),
                vmax=self.tensors["vp_true"].max(),
                cmap="seismic",
            ),
        )
        self.plot_loss()
        print(
            f"self.address bottom of plot_data={id(self)}",
            flush=True,
        )
        print(
            f"self.tensors.keys() at bottom of plot_data={self.tensors.keys()}",
            flush=True,
        )


if __name__ == "__main__":
    me = MultiscaleExample(
        data_save="dw/data",
        fig_save="dw/figs",
        pickle_save="dw/pickle",
        verbose=2,
        tensor_names=[
            "vp_true",
            "vp_init",
            "vp_record",
            "freqs",
            "loss",
            "src_amp_y",
            "src_loc_y",
            "rec_loc_y",
            "obs_data",
        ],
    )
    me.n_epochs = 2
    print(f"address Multiscale tensor keys: {me.tensors.keys()}")
    print(f"Multiscale address before run()={id(me)}")
    me = me.run()
    print(f"address MultiScale tensor keys: {me.tensors.keys()}")
    print(f"Multiscale address after run()={id(me)}")
