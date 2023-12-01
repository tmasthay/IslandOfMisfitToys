from ...utils import taper

from masthay_helpers.global_helpers import (
    DotDict,
    iraise,
    get_print,
)

from abc import ABC, abstractmethod
from itertools import product
import torch
from scipy.signal import butter
from torchaudio.functional import biquad
import numpy as np


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
        self.__setup_trainable_records(self.custom.loss_reshape)

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

    def record_trainables(self, idx):
        params = {
            k: v
            for k, v in self.dist_prop.module.named_parameters()
            if v.requires_grad
        }
        for k, v in params.items():
            key = k.replace('.p', '')
            print(self.report.keys(), flush=True)
            data = getattr(self.dist_prop.module, key)().detach().cpu()
            self.report.get(key + '_record')[idx] = data

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
        self.report.loss.append(loss_local.detach())

        if not other_info:
            return

        for k, v in other_info.items():
            if k not in self.report.keys():
                self.report.set(k, [])
            self.report.get(k).append(v.detach())

    def reset_optimizer(self):
        opt_type = type(self.optimizer)
        self.optimizer = opt_type(
            self.dist_prop.module.parameters(), **self.optimizer.defaults
        )

    def __setup_trainable_records(self, shape):
        for k, v in self.dist_prop.module.named_parameters():
            if v.requires_grad:
                key = k.replace('.p', '')
                self.report.set(
                    key + '_record', torch.zeros(*shape, *v.shape).to(self.rank)
                )
                print(self.report.keys())


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
                6,
                value,
                fs=1.0 / self.dist_prop.module.metadata.dt,
                output="sos",
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
                obj.dist_prop.module.obs_data.detach()
            )

            obj.report.obs_data_filt_record.append(self.custom.obs_data_filt)

            # This is the point where we might need to reset the optimizer in
            #   case there is something I am missing!
            self.reset_optimizer()

        def freq_postprocess(*, obj, path, combos, combo_num, field_num):
            pass

        def epoch_preprocess(*, obj, path, combos, combo_num, field_num):
            pass

        def epoch_postprocess(*, obj, path, combos, combo_num, field_num):
            self.print(f"epoch_postprocess", verbose=2)
            idx = combos["idx"][combo_num]
            obj.record_trainables(idx)

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
        self.report.vp_init = self.dist_prop.module.vp_init


class TrainingVanilla(Training):
    def __init__(
        self,
        *,
        dist_prop,
        rank,
        world_size,
        optimizer,
        scheduler,
        loss,
        n_epochs,
        loss_scale=1.0,
        **kw,
    ):
        epoch_groups = self.build_epoch_groups(n_epochs=n_epochs)
        self.loss_scale = loss_scale
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

    def build_epoch_groups(self, *, n_epochs):
        def epoch_preprocess(*, obj, path, combos, combo_num, field_num):
            pass

        def epoch_postprocess(*, obj, path, combos, combo_num, field_num):
            idx = combos["idx"][combo_num]
            obj.report.vp_record[idx] = obj.dist_prop.module.vp().detach().cpu()

        epoch_groups = [
            dict(
                name="Epochs",
                values=range(n_epochs),
                preprocess=epoch_preprocess,
                postprocess=epoch_postprocess,
            )
        ]
        return epoch_groups

    def _step(self, path, **kw):
        if "msg" in kw:
            del kw["msg"]
        out = self.dist_prop(1, **kw)
        loss = self.loss_scale * self.loss(out[-1], self.report.obs_data)
        return loss, {"out_record": out[-1].detach().cpu()}

    def pre_train(self, *, path, **kw):
        self.report.out_record = []

        self.report.vp_record = torch.zeros(
            *self.custom.loss_reshape, *self.dist_prop.module.vp.p.shape
        ).to(self.rank)
        self.report.obs_data = self.dist_prop.module.obs_data

    def _post_train(self, *, path):
        self.report.loss = torch.tensor(self.report.loss).to(self.rank)

        self.report.out_record = torch.stack(self.report.out_record).to(
            self.rank
        )

        self.report.vp_true = self.dist_prop.module.vp_true
        self.report.vp_init = self.dist_prop.module.vp_init
