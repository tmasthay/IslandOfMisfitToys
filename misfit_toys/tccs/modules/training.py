from dataclasses import dataclass
from collections import OrderedDict
import torch
from masthay_helpers.global_helpers import (
    get_print,
    flip_dict,
    subdict,
    DotDict,
)
from torch.optim.lr_scheduler import ChainedScheduler
from misfit_toys.utils import load_all, save, cleanup, filt, taper
from typing import Protocol, Callable, Any


@dataclass
class Training:
    rank: int
    world_size: int
    prop: torch.nn.Module
    obs_data: torch.Tensor
    loss_fn: torch.nn.Module
    optimizer: list
    training_stages: OrderedDict
    report_spec: dict
    _step: Callable
    scheduler: list = None
    verbose: int = 1

    def __post_init__(self):
        self.optimizer_kwargs = self.optimizer
        self.optimizer = self.optimizer[0](
            self.prop.parameters(), **self.optimizer[1]
        )
        if self.scheduler:
            self.scheduler = ChainedScheduler(
                [curr[0](self.optimizer, **curr[1]) for curr in self.scheduler]
            )
        self.report_spec.setdefault('path', 'out/parallel')
        for k, v in self.report_spec.items():
            if k == 'path':
                continue
            defaults = {
                'update': lambda x: getattr(x, k),
                'reduce': torch.stack,
                'presave': torch.stack,
            }
            self.report_spec[k] = {**defaults, **v}
        self.report_spec_flip = flip_dict(
            subdict(self.report_spec, exclude=['path'])
        )
        self.report = DotDict({k: [] for k in self.report_spec.keys()})
        self.print, _ = get_print(_verbose=self.verbose)

    def _pre_train(self):
        pass

    def save_report(self):
        for k, v in self.report.items():
            if k in self.report_spec_flip['presave'].keys():
                print(f"Presaving {k}", flush=True)
                print(f"v={v}", flush=True)
                v = self.report_spec_flip['presave'][k](v)
            save(
                v, f'{k}_record', rank=self.rank, path=self.report_spec['path']
            )

    def reduce_report(self):
        block = ['path']
        block.extend([f'{k}_record' for k in block])
        for k in self.report.keys():
            if k in block:
                continue
            v = load_all(
                f'{k}_record',
                world_size=self.world_size,
                path=self.report_spec['path'],
            )
            reduce_key = k.replace('_record', '')
            if reduce_key in self.report_spec_flip['reduce'].keys():
                v = self.report_spec_flip['reduce'][reduce_key](v)
            else:
                v = torch.stack(v)
            save(v, f'{k}_record', rank='', path=self.report_spec['path'])

    def _post_train(self):
        self.save_report()
        torch.distributed.barrier()

        if self.rank == 0:
            self.reduce_report()
        torch.distributed.barrier()
        cleanup()

    def train(self):
        self._pre_train()
        self._train()
        self._post_train()

    def _train(self):
        self.__recursive_train(
            level_data=self.training_stages,
            depth=0,
            max_depth=len(self.training_stages),
        )

    def __recursive_train(self, *, level_data, depth=0, max_depth=0):
        if depth == max_depth:
            self.step()  # Main training logic
            return

        level_name, level_info = list(level_data.items())[depth]
        data, preprocess, postprocess = (
            level_info["data"],
            level_info["preprocess"],
            level_info["postprocess"],
        )

        idt = "    " * depth
        for item in data:
            self.print(f"{idt}Preprocessing {level_name} {item}", verbose=2)
            preprocess(self, item)  # Assuming preprocess takes 'self'

            self.__recursive_train(
                level_data=level_data, depth=depth + 1, max_depth=max_depth
            )

            self.print(f"{idt}Postprocessing {level_name} {item}", verbose=2)
            postprocess(self, item)

    def __step(self):
        self._step(self)

    def step(self):
        num_calls = 0

        def closure():
            nonlocal num_calls
            num_calls += 1
            self.optimizer.zero_grad()
            self.__step()
            if num_calls == 1:
                self.update_records()
            return self.loss

        self.optimizer.step(closure)
        if self.scheduler:
            self.scheduler.step()

    def update_records(self):
        for k in self.report_spec_flip['update'].keys():
            if k in ['path', 'path_record']:
                continue
            if k not in self.report.keys():
                raise ValueError(
                    f"Key {k} not in report,"
                    f" update.keys()={self.report_spec_flip['update'].keys()},"
                    f" report.keys()={self.report.keys()}"
                )
            self.report[k].append(self.report_spec_flip['update'][k](self))

    def reset_optimizer(self):
        self.optimizer = self.optimizer_kwargs[0](
            self.prop.parameters(), **self.optimizer_kwargs[1]
        )
