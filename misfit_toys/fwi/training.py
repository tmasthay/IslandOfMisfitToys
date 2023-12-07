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
from typing import Callable, Any
from abc import ABC, abstractmethod


# TODO: Consider using a Protocol here
#   and encapsulate _step, _pre_train, _post_train inside of it.
#   I do not believe that it is necessary, but if things get more complicated,
#   it may be worth it.
@dataclass
class TrainingAbstract(ABC):
    """
    Abstract base class for training modules.

    Attributes:
        rank (int): The rank of the training module.
        world_size (int): The total number of training modules.
        prop (torch.nn.Module): The model to be trained.
        obs_data (torch.Tensor): The input data for training.
        loss_fn (torch.nn.Module): The loss function for training.
        optimizer (list): The optimizer used for training.
        report_spec (dict): The specification for reporting training progress.
        scheduler (list): The scheduler used for adjusting learning rate during training.
        verbose (int): The verbosity level for printing training progress.
        override_post_train (bool): Flag indicating whether to override the default post-training logic.

    Methods:
        __post_init__(): Initialize the training module.
        _step(): Perform a single step of training.
        _build_training_stages(): Build the training stages for initialization.
        train(): Train the model.
        step(): Perform a single step of training.
        reset_optimizer(): Reset the optimizer.
        _pre_train(): Run logic before training.
        _train(): Run the training process.
        _post_train(): Run logic after training.
        _update_records(): Update the training progress records.
        _save_report(): Save the training progress report.
        _reduce_report(): Reduce the training progress report.
        _post_train_default(): Default post-training logic.
        __recursive_train(): Recursively run the training process.
        __post_train(): Perform post-training operations.
    """

    rank: int
    world_size: int
    prop: torch.nn.Module
    obs_data: torch.Tensor
    loss_fn: torch.nn.Module
    optimizer: list
    report_spec: dict
    scheduler: list = None
    verbose: int = 1
    override_post_train: bool = False

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
        self.training_stages = self._build_training_stages()

    @abstractmethod
    def _step(self) -> None:
        """Main stepping logic"""

    @abstractmethod
    def _build_training_stages(self) -> OrderedDict:
        """Define training stages for initializer"""

    def train(self):
        self._pre_train()
        self._train()
        self.__post_train()

    def step(self):
        num_calls = 0

        def closure():
            nonlocal num_calls
            num_calls += 1
            self.optimizer.zero_grad()
            self._step()
            if num_calls == 1:
                self._update_records()
            return self.loss

        self.optimizer.step(closure)
        if self.scheduler:
            self.scheduler.step()

    def reset_optimizer(self):
        self.optimizer = self.optimizer_kwargs[0](
            self.prop.parameters(), **self.optimizer_kwargs[1]
        )

    def _pre_train(self) -> None:
        """Logic to run before training"""
        pass

    def _train(self):
        self.__recursive_train(
            level_data=self.training_stages,
            depth=0,
            max_depth=len(self.training_stages),
        )

    def _post_train(self) -> None:
        """Logic to run after training"""
        pass

    def _update_records(self):
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

    def _save_report(self):
        for k, v in self.report.items():
            if k in self.report_spec_flip['presave'].keys():
                print(f"Presaving {k}", flush=True)
                print(f"v={v}", flush=True)
                v = self.report_spec_flip['presave'][k](v)
            save(
                v, f'{k}_record', rank=self.rank, path=self.report_spec['path']
            )

    def _reduce_report(self):
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

    def _post_train_default(self):
        self._save_report()
        torch.distributed.barrier()

        if self.rank == 0:
            self._reduce_report()
        torch.distributed.barrier()
        cleanup()

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

    def __post_train(self):
        self._post_train()
        if not self.override_post_train:
            self._post_train_default()


class Training(TrainingAbstract):
    """
    A class representing the training process.

    Attributes:
        rank (int): The rank of the training process.
        world_size (int): The total number of training processes.
        prop (torch.nn.Module): The model to be trained.
        obs_data (torch.Tensor): The input data for training.
        loss_fn (torch.nn.Module): The loss function used for training.
        optimizer (list): The optimizer used for training.
        report_spec (dict): The specification for reporting training progress.
        scheduler (list, optional): The learning rate scheduler used for training. Defaults to None.
        verbose (int, optional): The verbosity level. Defaults to 1.
        override_post_train (bool, optional): Whether to override the post-training step. Defaults to False.
    """

    rank: int
    world_size: int
    prop: torch.nn.Module
    obs_data: torch.Tensor
    loss_fn: torch.nn.Module
    optimizer: list
    report_spec: dict
    scheduler: list = None
    verbose: int = 1
    override_post_train: bool = False

    def __init__(
        self,
        *,
        rank: int,
        world_size: int,
        prop: torch.nn.Module,
        obs_data: torch.Tensor,
        loss_fn: torch.nn.Module,
        optimizer: list,
        report_spec: dict,
        scheduler: list = None,
        verbose: int = 1,
        override_post_train: bool = False,
        _step: Callable[[TrainingAbstract], None],
        _pre_train: Callable[[TrainingAbstract], None] = None,
        _post_train: Callable[[TrainingAbstract], None] = None,
        _build_training_stages: Callable[[TrainingAbstract], OrderedDict],
    ):
        """initializer"""
        self._step_helper = _step
        self._build_training_stages_helper = _build_training_stages
        self._pre_train_helper = _pre_train if _pre_train else lambda x: None
        self._post_train_helper = _post_train if _post_train else lambda x: None
        super().__init__(
            rank=rank,
            world_size=world_size,
            prop=prop,
            obs_data=obs_data,
            loss_fn=loss_fn,
            optimizer=optimizer,
            report_spec=report_spec,
            scheduler=scheduler,
            verbose=verbose,
            override_post_train=override_post_train,
        )

    def _step(self):
        self._step_helper(self)

    def _build_training_stages(self) -> OrderedDict:
        return self._build_training_stages_helper()

    def _pre_train(self):
        self._pre_train_helper(self)

    def _post_train(self):
        self._post_train_helper(self)
