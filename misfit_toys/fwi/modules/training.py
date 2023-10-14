import torch
from scipy.signal import butter
from torchaudio.functional import biquad
from ...utils import taper, canonical_tensors
import numpy as np
import torch.distributed as dist
from masthay_helpers.global_helpers import (
    call_counter,
    iprint,
    printj,
    DotDict,
    iraise,
    get_print,
    summarize_tensor,
    print_tensor,
)

from .distribution import cleanup

import os
from abc import ABC, abstractmethod
from itertools import product

from ..custom_losses import W1


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
        # loss_fn = W1(lambda x: x**2)

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
                6, cutoff_freq, fs=1.0 / self.dist_prop.module.dt, output="sos"
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

            s = summarize_tensor(observed_data_filt)
            print(f"observed_data_filt={s}", flush=True)

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


class Training1(ABC):
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
        reduce,
        prec=".16f",
        **kw,
    ):
        self.dist_prop = dist_prop
        self.rank = rank
        self.world_size = world_size
        self.print, self.printj = get_print(_verbose=verbose)
        self.prec = prec
        self.reduce = reduce

        # NOTE: if something goes wrong, this is the first place to check
        #     for a bug. In previous version, the optimizer was created
        #     in the preprocess method of the epochs group. This means
        #     that the optimizer was created every epoch. In this framework,
        #     we only generated ONCE at the very beginning!
        self.optimizer = optimizer[0](
            self.dist_prop.module.parameters(), **optimizer[1]
        )
        self.loss = loss

        self.scheduler = self.__build_scheduler(scheduler)
        self.epoch_groups = self.__build_epoch_groups(epoch_groups)
        self.combos = self.__build_combos()

        self.custom = DotDict(kw)
        self.report = DotDict({"loss": []})
        self.custom.loss_reshape = [len(e["values"]) for e in epoch_groups]

    @abstractmethod
    def _step(self, path, **kw):
        pass

    def train(self, *, path, **kw):
        self.pre_train(path=path, **kw)
        self._train(path=path, **kw)
        self.post_train(path=path, **kw)

    def pre_train(self, *, path, **kw):
        pass

    def step(self, path, **kw):
        loss = 0.0
        calls = 0

        def closure():
            nonlocal loss, calls
            calls += 1
            self.optimizer.zero_grad()
            loss_local, other_info = self._step(path=path, **kw)
            if calls == 1:
                loss = loss_local
                self.__update_step_info(loss_local, other_info)
            loss_local.backward()
            return loss_local

        self.optimizer.step(closure)
        if self.scheduler is not None:
            self.scheduler.step()
        return loss

    def post_train(self, *, path, **kw):
        self._post_train(path=path, **kw)

    def _post_train(self, *, path, **kw):
        pass

    def _train(self, *, path, **kw):
        combos = self.combos

        def get_msg(combo):
            cols = []
            for name, val in zip(combos["names"], combo):
                s = f"{name}="
                if isinstance(val, float):
                    s += f"{val:{self.prec}}"
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
                    combos=combos,
                    combo_num=j,
                    field_num=i,
                    **kw,
                )
            loss = self.step(path=path, msg=get_msg(combo), **kw)
            msg = get_msg(combo)
            msg += f" & Loss={loss:{self.prec}}"
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

    def __build_scheduler(self, scheduler):
        if scheduler is None:
            return None
        schedule_list = []
        for s in scheduler:
            schedule_list.append(s[0](self.optimizer, **s[1]))
        scheduler = torch.optim.lr_scheduler.ChainedScheduler(schedule_list)
        return scheduler

    def __build_epoch_groups(self, epoch_groups):
        for i, e in enumerate(epoch_groups):
            try:
                epoch_groups[i]["values"] = torch.tensor(
                    e["values"], dtype=type(e["values"][0])
                )
            except:
                iraise(
                    ValueError,
                    f"Unknown type {type(e['values'])} passed to",
                    f" epoch_groups[{i}]['values']. ",
                    f"Must conform to torch.Tensor constructor.",
                )
        return epoch_groups

    def __build_combos(self):
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
        return combos

    def __update_step_info(self, loss_local, other_info):
        self.report.loss.append(loss_local.detach().cpu())

        if not other_info:
            return

        for k, v in other_info.items():
            if k not in self.report.keys():
                self.report.set(k, [])
            self.report.get(k).append(v.detach().cpu())


class TrainingMultiscale(Training1):
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
        epoch_groups = self.build_epoch_groups(freqs=freqs, n_epochs=n_epochs)
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

    def pre_train(self, *, path, **kw):
        self.report.obs_data_filt_record = []
        self.report.out_record = []
        self.report.out_filt_record = []

        self.report.vp_record = torch.zeros(
            *self.custom.loss_reshape, *self.dist_prop.module.vp.p.shape
        )

        self.report.obs_data = self.dist_prop.module.obs_data.detach().cpu()

    def build_epoch_groups(self, *, freqs, n_epochs):
        def freq_preprocess(*, obj, path, combos, combo_num, field_num):
            self.print(f"freq_preprocess", verbose=2)
            value = combos["values"][combo_num][field_num]
            sos = butter(
                6, value, fs=1.0 / self.dist_prop.module.dt, output="sos"
            )
            obj.custom.sos = [
                torch.tensor(sosi)
                .to(self.dist_prop.module.obs_data.dtype)
                .to(self.rank)
                for sosi in sos
            ]
            obj.custom.filt = lambda x: biquad(
                biquad(biquad(x, *obj.custom.sos[0]), *obj.custom.sos[1]),
                *obj.custom.sos[2],
            )
            obj.custom.obs_data_filt = obj.custom.filt(
                obj.dist_prop.module.obs_data
            )

            obj.report.obs_data_filt_record.append(
                self.custom.obs_data_filt.detach().cpu()
            )

            # s = summarize_tensor(obj.custom.obs_data_filt)
            # self.print(f"obs_data_filt={s}", verbose=1)

            # This is the point where we might need to reset the optimizer in
            #   case there is something I am missing!
            self.optimizer = torch.optim.LBFGS(
                self.dist_prop.module.parameters()
            )

        def freq_postprocess(*, obj, path, combos, combo_num, field_num):
            pass

        def epoch_preprocess(*, obj, path, combos, combo_num, field_num):
            # obj.custom.obs_data_filt = obj.custom.filt(
            #     obj.dist_prop.module.obs_data
            # )
            self.print(f"epoch_preprocess", verbose=2)
            pass

        def epoch_postprocess(*, obj, path, combos, combo_num, field_num):
            self.print(f"epoch_postprocess", verbose=2)
            idx = combos["idx"][combo_num]
            obj.report.vp_record[idx] = obj.dist_prop.module.vp().detach().cpu()

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

    def _step(self, path, **kw):
        if "msg" in kw:
            del kw["msg"]
        out = self.dist_prop(1, **kw)
        out_filt = self.custom.filt(taper(out[-1], 100))
        loss = 1e6 * self.loss(out_filt, self.custom.obs_data_filt)
        return loss, {
            "out_record": out[-1].detach().cpu(),
            "out_filt_record": out_filt.detach().cpu(),
        }

    def _post_train(self, *, path):
        self.report.out_init = self.report.out_record[0]
        self.report.out_true = self.report.out_record[-1]
        self.report.out_filt_init = self.report.out_filt_record[0]
        self.report.out_filt_true = self.report.out_filt_record[-1]
        self.report.obs_data_filt_init = self.report.obs_data_filt_record[0]
        self.report.obs_data_filt_true = self.report.obs_data_filt_record[-1]

        n_shots, n_rec, nt = self.dist_prop.module.obs_data.shape
        self.report.loss = torch.tensor(self.report.loss).reshape(
            self.custom.loss_reshape
        )
        self.report.obs_data_filt_record = torch.stack(
            self.report.obs_data_filt_record
        ).reshape(n_shots, n_rec, nt, *self.custom.loss_reshape)
        self.report.out_record = torch.stack(self.report.out_record).reshape(
            n_shots, n_rec, nt, *self.custom.loss_reshape
        )
        self.report.out_filt_record = torch.stack(
            self.report.out_filt_record
        ).reshape(n_shots, n_rec, nt, *self.custom.loss_reshape)

        self.report.freqs = torch.Tensor([10, 15, 20, 25, 30])
        self.report.vp_true = self.dist_prop.module.vp_true.detach().cpu()


class Training1(ABC):
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
        prec=".16f",
        **kw,
    ):
        self.dist_prop = dist_prop
        self.rank = rank
        self.world_size = world_size
        self.print, self.printj = get_print(_verbose=verbose)
        self.prec = prec

        # NOTE: if something goes wrong, this is the first place to check
        #     for a bug. In previous version, the optimizer was created
        #     in the preprocess method of the epochs group. This means
        #     that the optimizer was created every epoch. In this framework,
        #     we only generated ONCE at the very beginning!
        self.optimizer = optimizer[0](
            self.dist_prop.module.parameters(), **optimizer[1]
        )
        self.loss = loss

        self.scheduler = self.__build_scheduler(scheduler)
        self.epoch_groups = self.__build_epoch_groups(epoch_groups)
        self.combos = self.__build_combos()

        self.custom = DotDict(kw)
        self.report = DotDict({"loss": []})
        self.custom.loss_reshape = [len(e["values"]) for e in epoch_groups]

    @abstractmethod
    def _step(self, path, **kw):
        pass

    def train(self, *, path, **kw):
        self.pre_train(path=path, **kw)
        self._train(path=path, **kw)
        self.post_train(path=path, **kw)

    def pre_train(self, *, path, **kw):
        pass

    def step(self, path, **kw):
        loss = 0.0
        calls = 0

        def closure():
            nonlocal loss, calls
            calls += 1
            self.optimizer.zero_grad()
            loss_local, other_info = self._step(path=path, **kw)
            if calls == 1:
                loss = loss_local
                self.__update_step_info(loss_local, other_info)
            loss_local.backward()
            return loss_local

        self.optimizer.step(closure)
        if self.scheduler is not None:
            self.scheduler.step()
        return loss

    def post_train(self, *, path, **kw):
        self._post_train(path=path, **kw)

    def _post_train(self, *, path, **kw):
        pass

    def _train(self, *, path, **kw):
        combos = self.combos

        def get_msg(combo):
            cols = []
            for name, val in zip(combos["names"], combo):
                s = f"{name}="
                if isinstance(val, float):
                    s += f"{val:{self.prec}}"
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
                    combos=combos,
                    combo_num=j,
                    field_num=i,
                    **kw,
                )
            loss = self.step(path=path, msg=get_msg(combo), **kw)
            msg = get_msg(combo)
            msg += f" & Loss={loss:{self.prec}}"
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

    def __build_scheduler(self, scheduler):
        if scheduler is None:
            return None
        schedule_list = []
        for s in scheduler:
            schedule_list.append(s[0](self.optimizer, **s[1]))
        scheduler = torch.optim.lr_scheduler.ChainedScheduler(schedule_list)
        return scheduler

    def __build_epoch_groups(self, epoch_groups):
        for i, e in enumerate(epoch_groups):
            try:
                epoch_groups[i]["values"] = torch.tensor(
                    e["values"], dtype=type(e["values"][0])
                )
            except:
                iraise(
                    ValueError,
                    f"Unknown type {type(e['values'])} passed to",
                    f" epoch_groups[{i}]['values']. ",
                    f"Must conform to torch.Tensor constructor.",
                )
        return epoch_groups

    def __build_combos(self):
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
        return combos

    def __update_step_info(self, loss_local, other_info):
        self.report.loss.append(loss_local.detach().cpu())

        if not other_info:
            return

        for k, v in other_info.items():
            if k not in self.report.keys():
                self.report.set(k, [])
            self.report.get(k).append(v.detach().cpu())


class TrainingMultiscale(Training1):
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
        epoch_groups = self.build_epoch_groups(freqs=freqs, n_epochs=n_epochs)
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

    def build_epoch_groups(self, *, freqs, n_epochs):
        def freq_preprocess(*, obj, path, combos, combo_num, field_num):
            self.print(f"freq_preprocess", verbose=2)
            value = combos["values"][combo_num][field_num]
            sos = butter(
                6, value, fs=1.0 / self.dist_prop.module.dt, output="sos"
            )
            obj.custom.sos = [
                torch.tensor(sosi)
                .to(self.dist_prop.module.obs_data.dtype)
                .to(self.rank)
                for sosi in sos
            ]
            obj.custom.filt = lambda x: biquad(
                biquad(biquad(x, *obj.custom.sos[0]), *obj.custom.sos[1]),
                *obj.custom.sos[2],
            )
            obj.custom.obs_data_filt = obj.custom.filt(
                obj.dist_prop.module.obs_data
            )

            obj.report.obs_data_filt_record.append(
                self.custom.obs_data_filt.detach().cpu()
            )

            # s = summarize_tensor(obj.custom.obs_data_filt)
            # self.print(f"obs_data_filt={s}", verbose=1)

            # This is the point where we might need to reset the optimizer in
            #   case there is something I am missing!
            self.optimizer = torch.optim.LBFGS(
                self.dist_prop.module.parameters()
            )

        def freq_postprocess(*, obj, path, combos, combo_num, field_num):
            pass

        def epoch_preprocess(*, obj, path, combos, combo_num, field_num):
            # obj.custom.obs_data_filt = obj.custom.filt(
            #     obj.dist_prop.module.obs_data
            # )
            pass

        def epoch_postprocess(*, obj, path, combos, combo_num, field_num):
            self.print(f"epoch_postprocess", verbose=2)
            idx = combos["idx"][combo_num]
            obj.report.vp_record[idx] = obj.dist_prop.module.vp().detach().cpu()

        epoch_groups = [
            dict(
                name="Frequencies",
                values=freqs,
                preprocess=freq_preprocess,
                postprocess=freq_postprocess,
            ),
            dict(
                name="Epochs",
                values=range(n_epochs),
                preprocess=epoch_preprocess,
                postprocess=epoch_postprocess,
            ),
        ]
        return epoch_groups

    def _step(self, path, **kw):
        if "msg" in kw:
            del kw["msg"]
        out = self.dist_prop(1, **kw)
        out_filt = self.custom.filt(taper(out[-1], 100))
        loss = 1e6 * self.loss(out_filt, self.custom.obs_data_filt)
        return loss, {
            "out_record": out[-1].detach().cpu(),
            "out_filt_record": out_filt.detach().cpu(),
        }

    def pre_train(self, *, path, **kw):
        self.report.obs_data_filt_record = []
        self.report.out_record = []
        self.report.out_filt_record = []

        self.report.vp_record = torch.zeros(
            *self.custom.loss_reshape, *self.dist_prop.module.vp.p.shape
        ).to(self.rank)

    def _post_train(self, *, path):
        self.report.obs_data = self.dist_prop.module.obs_data
        n_shots, n_rec, nt = self.report.obs_data.shape

        self.report.loss = (
            torch.tensor(self.report.loss)
            .reshape(self.custom.loss_reshape)
            .to(self.rank)
        )
        self.report.obs_data_filt_record = (
            torch.stack(self.report.obs_data_filt_record)
            .reshape(n_shots, n_rec, nt, *self.custom.loss_reshape)
            .to(self.rank)
        )
        self.report.out_record = (
            torch.stack(self.report.out_record)
            .reshape(n_shots, n_rec, nt, *self.custom.loss_reshape)
            .to(self.rank)
        )
        self.report.out_filt_record = (
            torch.stack(self.report.out_filt_record)
            .reshape(n_shots, n_rec, nt, *self.custom.loss_reshape)
            .to(self.rank)
        )

        self.report.freqs = torch.Tensor([10, 15, 20, 25, 30])
        self.report.vp_true = self.dist_prop.module.vp_true
