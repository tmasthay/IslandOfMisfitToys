import torch
from scipy.signal import butter
from torchaudio.functional import biquad
from ...utils import taper, summarize_tensor, print_tensor
import numpy as np
import torch.distributed as dist
from masthay_helpers import call_counter

from .distribution import cleanup

import os
from abc import ABC, abstractmethod
from itertools import product


class TrainingDummy(ABC):
    def __init__(self, *, dist_prop, rank, world_size):
        self.dist_prop = dist_prop
        self.rank = rank
        self.world_size = world_size

    @abstractmethod
    def train(self, *, path, **kw):
        pass


class TrainingMultiscale(TrainingDummy):
    def __init__(self, *, dist_prop, rank, world_size):
        self.dist_prop = dist_prop
        self.rank = rank
        self.world_size = world_size

    def train(self, *, path, **kw):
        # Setup optimiser to perform inversion
        loss_fn = torch.nn.MSELoss()

        # Run optimisation/inversion
        n_epochs = 2

        all_freqs = torch.Tensor([10, 15, 20, 25, 30])
        n_freqs = all_freqs.shape[0]
        freqs = all_freqs
        loss_local = torch.zeros(freqs.shape[0], n_epochs).to(self.rank)
        vp_record = torch.Tensor(
            n_freqs, n_epochs, *self.dist_prop.module.vp.p.shape
        )

        print(
            f"enumerate(all_freq)={[e for e in enumerate(all_freqs)]}",
            flush=True,
        )
        for idx, cutoff_freq in enumerate(list(all_freqs)):
            sos = butter(
                6,
                cutoff_freq,
                fs=1.0 / self.dist_prop.module.dt,
                output="sos",
            )
            sos = [
                torch.tensor(sosi)
                .to(self.dist_prop.module.obs_data.dtype)
                .to(self.rank)
                for sosi in sos
            ]
            # input(sos)

            def filt(x):
                return biquad(biquad(biquad(x, *sos[0]), *sos[1]), *sos[2])

            observed_data_filt = filt(self.dist_prop.module.obs_data)

            optimiser = torch.optim.LBFGS(self.dist_prop.module.parameters())

            # print_tensor(observed_data_filt, print_fn=input)

            for epoch in range(n_epochs):
                closure_calls = 0

                def closure():
                    nonlocal closure_calls, loss_local
                    closure_calls += 1
                    optimiser.zero_grad()
                    # out = self.distribution.dist_prop.module(**kw)
                    out = self.dist_prop(1, **kw)
                    out_filt = filt(taper(out[-1], 100))
                    loss = 1e6 * loss_fn(out_filt, observed_data_filt)
                    if closure_calls == 1:
                        print(
                            (
                                f"Loss={loss.item():.16f}, "
                                f"Freq={cutoff_freq}, "
                                f"Epoch={epoch}, "
                                f"Rank={self.rank}"
                            ),
                            flush=True,
                        )
                        loss_local[idx, epoch] = loss
                    loss.backward()
                    return loss

                optimiser.step(closure)
                vp_record[idx, epoch] = (
                    self.dist_prop.module.vp().detach().cpu()
                )
        os.makedirs(path, exist_ok=True)

        def save(k, v):
            u = v.detach().cpu()
            lcl_path = os.path.join(path, f"{k}_{self.rank}.pt")
            print(f"Saving to {lcl_path}...", flush=True, end="")
            torch.save(u, lcl_path)
            print("SUCCESS", flush=True)

        save("loss", loss_local)
        save("freqs", freqs)
        save("vp_record", vp_record)


class Training(ABC):
    def __init__(self, *, dist_prop, rank, world_size, verbose=1):
        self.dist_prop = dist_prop
        self.rank = rank
        self.world_size = world_size
        self.print = self.get_print(verbose)

    def train(self, *, path, epoch_idx, **kw):
        self.pre_train(path=path, **kw)
        self._train(path=path, epoch_idx=epoch_idx, **kw)
        self.post_train(path=path, **kw)

    @staticmethod
    def get_print(_verbose):
        def print_fn(*args, verbose, **kw):
            if verbose <= _verbose:
                kw["flush"] = True
                print(*args, **kw)

        return print_fn

    def pre_train(self, *, path, **kw):
        pass

    def _train(self, *, path, **kw):
        epoch_groups = kw["epoch_groups"]
        epoch_values = [e["values"] for e in epoch_groups]
        epoch_names = [e["name"] for e in epoch_groups]

        for combo in product(*epoch_groups):
            msgs = [f'{e["name"]}={e["value"]}' for e in combo]
            preprocess_vals = [
                e["preprocess"](obj=self, path=path, **kw) for e in combo
            ]
            train_vals = self.step(
                path=path, preprocess_vals=preprocess_vals, **kw
            )
            postprocess_vals = [
                e["postprocess"](path=path, train_vals=train_vals, **kw)
                for e in combo
            ]
        return train_vals, postprocess_vals

    def step(self, path, preprocess_vals=None, **kw):
        loss = 0.0

        @call_counter(_verbose=1)
        def closure():
            nonlocal loss
            self.optimizer.zero_grad()
            loss_local = self._step(
                path=path, preprocess_vals=preprocess_vals, **kw
            )
            if closure.calls == 1:
                loss = loss_local
                self.print(f"Loss={loss:.16f}", verbose=closure.verbose)
            loss_local.backward()
            return loss_local

        self.optimizer.step(closure)
        self.scheduler.step()
        return loss

    @abstractmethod
    def _step(self, path, preprocess_vals=None, **kw):
        pass
