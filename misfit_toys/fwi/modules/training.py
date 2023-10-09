import torch
from scipy.signal import butter
from torchaudio.functional import biquad
from ...utils import taper, summarize_tensor, print_tensor
import numpy as np
import torch.distributed as dist
from masthay_helpers import call_counter, iprint, printj, DotDict

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


class TrainingMultiscaleLegacy(TrainingDummy):
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
    def __init__(
        self,
        *,
        dist_prop,
        rank,
        world_size,
        verbose=1,
        optimizer,
        scheduler,
        loss,
        epoch_groups,
        **kw,
    ):
        self.dist_prop = dist_prop
        self.rank = rank
        self.world_size = world_size
        self.print, self.printj = self.get_print(verbose)

        # NOTE: if something goes wrong, this is the first place to check
        #     for a bug. In previous version, the optimizer was created
        #     in the preprocess method of the epochs group. This means
        #     that the optimizer was created every epoch. In this framework,
        #     we only generated ONCE at the very beginning!
        self.optimizer = optimizer[0](
            self.dist_prop.module.parameters(), **optimizer[1]
        )
        self.scheduler = self.__build_scheduler(scheduler)
        self.loss = loss
        self.epoch_groups = epoch_groups
        self.custom = kw

    def train(self, *, path, epoch_idx, **kw):
        self.pre_train(path=path, **kw)
        self._train(path=path, epoch_idx=epoch_idx, **kw)
        self.post_train(path=path, **kw)

    def __build_scheduler(self, scheduler):
        schedule_list = []
        for s in scheduler:
            schedule_list.append(s[0](self.optimizer, **s[1]))
        self.scheduler = torch.optim.lr_scheduler.ChainedScheduler(
            schedule_list
        )

    def getc(self, key):
        return self.custom.get(key)

    def setc(self, key, val):
        self.custom.set(key, val)

    @staticmethod
    def get_print(
        _verbose,
        l=0,
        idt_str="    ",
        cpl=80,
        sep="\n",
        mode="auto",
        demarcator="&",
        align="ljust",
        extra_space=0,
    ):
        def print_fn(*args, verbose=1, **kw):
            if verbose <= _verbose:
                kw["flush"] = True
                iprint(*args, l=l, idt_str=idt_str, cpl=cpl, sep=sep, **kw)

        def print_col_fn(*args, verbose=1, **kw):
            if verbose <= _verbose:
                kw["flush"] = True
                printj(
                    args,
                    demarcator=demarcator,
                    align=align,
                    extra_space=extra_space,
                    **kw,
                )
            return print

        return print_fn, print_col_fn

    def pre_train(self, *, path, **kw):
        pass

    def _train(self, *, path, **kw):
        epoch_groups = self.epoch_groups
        combos = dict(
            values=tuple(product(*[e["values"] for e in epoch_groups])),
            idx=tuple(
                product(*[range(len(e["values"])) for e in epoch_groups])
            ),
            names=[e["name"] for e in epoch_groups],
            preprocess=[e["preprocess"] for e in epoch_groups],
            postprocess=[e["postprocess"] for e in epoch_groups],
        )

        def get_msg(combo):
            cols = []
            for name, val in zip(combos["names"], combo):
                s = f'{combos["name"]}='
                if isinstance(val, float):
                    s += f"{val:.4e}"
                else:
                    s += f"{val}"
                cols.append(s)
            cols.append(f"Rank={self.rank}")
            msg = " & ".join(cols)
            return msg

        for j, combo in enumerate(combos["values"]):
            for i, e in enumerate(combos["preprocess"]):
                e(
                    obj=self,
                    path=path,
                    value=combos,
                    combo_num=j,
                    field_num=i,
                    **kw,
                )
            loss = self.step(path=path, msg=get_msg(combo), **kw)
            msg = get_msg(combo)
            msg += f" & Loss={loss:.4e}"
            for i, e in enumerate(combos["postprocess"]):
                e(
                    obj=self,
                    path=path,
                    combos=combos,
                    combo_num=j,
                    field_num=i,
                    **kw,
                )
            self.printj(msg, verbose=1)
        return loss

    def step(self, path, **kw):
        loss = 0.0

        @call_counter(_verbose=1)
        def closure():
            nonlocal loss
            self.optimizer.zero_grad()
            loss_local = self._step(path=path, **kw)
            if closure.calls == 1:
                loss = loss_local
            loss_local.backward()
            return loss_local

        self.optimizer.step(closure)
        self.scheduler.step()
        return loss

    @abstractmethod
    def _step(self, path, **kw):
        pass

    def save_customs(self, *, path, keys):
        for k in keys():
            if k not in self.custom.keys():
                raise ValueError(
                    f"Key {k} not in self.custom.keys()={self.custom.keys()}"
                )
            self.print(f"Saving {k}...", verbose=1, end="")
            torch.save(self.custom.get(k), os.path.join(path, f"{k}.pt"))
            self.print("SUCCESS", verbose=1)

    def post_train(self, *, path, **kw):
        pass


class TrainingMultiscale(Training):
    def __init__(
        self,
        *,
        dist_prop,
        rank,
        world_size,
        optimizer,
        scheduler,
        loss,
        freqs,
        n_epochs,
        **kw,
    ):
        epoch_groups = self.build_epoch_groups(freqs, n_epochs)
        super().__init__(
            dist_prop=dist_prop,
            rank=rank,
            world_size=world_size,
            optimizer=optimizer,
            scheduler=scheduler,
            loss=loss,
            epoch_groups=epoch_groups,
            **kw,
        )

    def pre_train(self, *, path):
        assert len(self.epoch_groups) == 2
        self.custom.vp_record = torch.Tensor(
            self.epoch_groups[0]["values"].shape[0],
            self.epoch_groups[1]["values"].shape[0],
            *self.dist_prop.module.vp.p.shape,
        )

    def _step(self, path, **kw):
        out = self.dist_prop(1, **kw)
        out_filt = self.custom.filt(taper(out[-1], 100))
        loss = 1e6 * self.loss(out_filt, self.custom.obs_data_filt)

    def post_train(self, *, path):
        self.save_customs(path=path, keys=["vp_record"])

    def build_epoch_groups(self, *, freqs, n_epochs):
        def freq_preprocess(*, obj, path, combos, combo_num, field_num):
            value = combos["values"][combo_num][field_num]
            sos = butter(
                6,
                value,
                fs=1.0 / self.dist_prop.module.dt,
                output="sos",
            )
            obj.custom.sos = [
                torch.tensor(sosi)
                .to(self.dist_prop.module.obs_data.dtype)
                .to(self.rank)
                for sosi in sos
            ]
            obj.custom.filt = (
                lambda x: biquad(
                    biquad(biquad(x, *obj.custom.sos[0]), *obj.custom.sos[1]),
                    *obj.custom.sos[2],
                ),
            )
            obj.custom.obs_data_filt = obj.custom.filt(
                obj.dist_prop.module.obs_data
            )

            # This is the point where we might need to reset the optimizer in
            #   case there is something I am missing!
            # self.optimizer = torch.optim.LBFGS(self.dist_prop.module.parameters())

        def freq_postprocess(*, obj, path, combos, combo_num, field_num):
            pass

        def epoch_preprocess(*, obj, path, combos, combo_num, field_num):
            obj.custom.obs_data_filt = obj.custom.filt(
                obj.dist_prop.module.obs_data
            )

        def epoch_postprocess(*, obj, path, combos, combo_num, field_num):
            idx = combos["idx"][combo_num]
            obj.custom.vp_record[idx] = obj.dist_prop.module.vp().detach().cpu()

        epoch_groups = []
        epoch_groups.append(
            dict(
                name="Frequencies",
                values=freqs,
                preprocess=freq_preprocess,
                postprocess=freq_postprocess,
            )
        )
        epoch_groups.append(
            dict(
                name="Epochs",
                values=range(n_epochs),
                preprocess=epoch_preprocess,
                postprocess=epoch_postprocess,
            )
        )
        return epoch_groups
