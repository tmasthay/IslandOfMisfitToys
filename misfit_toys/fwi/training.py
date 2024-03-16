"""Training module for FWI

Classes:
    TrainingAbstract: Abstract base class for training modules.
    Training: Subclass of TrainingAbstract where abstract methods supplied in __init__.

"""

from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable

import torch
from mh.core import DotDict, flip_dict, get_print
from mh.core_legacy import subdict
from torch.optim.lr_scheduler import ChainedScheduler

from misfit_toys.utils import cleanup, filt, load_all, save, taper

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
        """
        Perform post-initialization tasks.

        Parameters:
            None

        Returns:
            None

        Note:
            This method initializes the following attributes:
            - `self.optimizer_kwargs`: The optimizer keyword arguments.
            - `self.optimizer`: The optimizer instance.
            - `self.scheduler`: The scheduler instance.
            - `self.report_spec`: The report specification dictionary.
            - `self.report_spec_flip`: The flipped report specification dictionary.
            - `self.report`: The report dictionary.
            - `self.print`: The print function.
            - `self.training_stages`: The training stages list.
        """
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
            subdict(self.report_spec, exc=['path'])
        )
        self.report = DotDict({k: [] for k in self.report_spec.keys()})
        self.report.path = self.report_spec['path']
        self.print, _ = get_print(_verbose=self.verbose)
        self.training_stages = self._build_training_stages()

    def __set_loss_status(self, val):
        if hasattr(self.loss_fn, 'status'):
            self.loss_fn.build_status = val

    def _report_iteration(self):
        """
        Report the current iteration of training.

        Returns:
            None
        """
        if hasattr(self.loss_fn, 'status'):
            s = self.loss_fn.status
        else:
            s = (
                f"rank: {self.rank}, "
                f"iter: {len(self.report['loss'])}, "
                f"loss: {self.loss}"
            )
        out_norm = (
            self.out[-1].norm() if type(self.out) is tuple else self.out.norm()
        )
        s += (
            f", training.loss: {self.loss:.2e}"
            f", lr: {self.optimizer.param_groups[0]['lr']:.3e}"
            f", obs_data.norm: {self.obs_data.norm():.2e}"
            f", out.norm: {out_norm:.2e}"
            f", rank: {self.rank}"
        )
        self.print(s, verbose=1)

    @abstractmethod
    def _step(self) -> None:
        """Main stepping logic"""

    @abstractmethod
    def _build_training_stages(self) -> OrderedDict:
        """Define training stages for initializer"""

    def train(self):
        """
        Trains the model by performing pre-training, training, and post-training steps.
        """
        self._pre_train()
        self._train()
        self.__post_train()
        # torch.distributed.barrier()

    def step(self):
        """
        Performs a single optimization step.

        Returns:
            loss (float): The loss value after the optimization step.
        """
        num_calls = 0

        def closure():
            nonlocal num_calls
            num_calls += 1
            self.optimizer.zero_grad()

            self.__set_loss_status(num_calls == 1)

            self._step()
            if num_calls == 1:
                self._update_records()
                self._report_iteration()
            return self.loss

        self.optimizer.step(closure)
        if self.scheduler:
            self.scheduler.step()
        return self.loss

    def reset_optimizer(self):
        """
        Resets the optimizer used for parameter updates.

        Returns:
            None
        """
        self.optimizer = self.optimizer_kwargs[0](
            self.prop.parameters(), **self.optimizer_kwargs[1]
        )

    def _pre_train(self) -> None:
        """Logic to run before training

        Returns:
            None
        """
        pass

    def _train(self):
        """
        Recursively trains the model using the specified training stages.

        Returns:
            None
        """
        self.__recursive_train(
            level_data=self.training_stages,
            depth=0,
            max_depth=len(self.training_stages),
        )

    def _post_train(self) -> None:
        """Logic to run after training

        Returns:
            None
        """
        pass

    def _update_records(self):
        """
        Update the records in the report.

        Returns:
            None
        """
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
        """
        Save the report items to files.

        This method iterates over the report items and saves them to files using the specified paths.
        Before saving, it checks if any pre-save functions are defined for the item and applies them if necessary.

        Returns:
            None
        """
        # tmp = {k: v.shape for k, v in self.report}
        # raise ValueError(f'My report = {self.report.path}')
        for k, v in self.report.items():
            if k in self.report_spec_flip['presave'].keys():
                print(f"Presaving {k}", flush=True)
                # print(f"v={v}", flush=True)
                try:
                    v = self.report_spec_flip['presave'][k](v)
                except Exception as e:
                    msg = f"Error in presave for k: v={k}: {v}"
                    raise ValueError(f'{e}\n\n{msg}')
                # raise ValueError(f"v={v}")
            save(
                v, f'{k}_record', rank=self.rank, path=self.report_spec['path']
            )

    def _reduce_report(self):
        """
        Reduce the report by loading and reducing the records for each key in the report.
        The reduced records are then saved back to the report.

        Returns:
            None
        """
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
        """
        Perform post-training operations.

        This method saves the training report, reduces the report on rank 0,
        and performs cleanup.

        """
        self._save_report()
        torch.distributed.barrier()

        if self.rank == 0:
            self._reduce_report()
        torch.distributed.barrier()
        cleanup()

    def __recursive_train(self, *, level_data, depth=0, max_depth=0):
        """
        Recursively trains the model on different levels of data.

        Args:
            level_data (dict): A dictionary containing the level data.
            depth (int): The current depth of recursion.
            max_depth (int): The maximum depth of recursion.

        Returns:
            None
        """
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
        """
        Initialize the Training object.

        Args:
            rank (int): The rank of the current process.
            world_size (int): The total number of processes.
            prop (torch.nn.Module): The model to be trained.
            obs_data (torch.Tensor): The observed data.
            loss_fn (torch.nn.Module): The loss function.
            optimizer (list): The optimizer(s) to be used.
            report_spec (dict): The specification for reporting training progress.
            scheduler (list, optional): The scheduler(s) to be used. Defaults to None.
            verbose (int, optional): The verbosity level. Defaults to 1.
            override_post_train (bool, optional): Whether to override the post-training step. Defaults to False.
            _step (Callable[[TrainingAbstract], None]): The step function for training.
            _pre_train (Callable[[TrainingAbstract], None], optional): The pre-training step function. Defaults to None.
            _post_train (Callable[[TrainingAbstract], None], optional): The post-training step function. Defaults to None.
            _build_training_stages (Callable[[TrainingAbstract], OrderedDict]): The function to build training stages.
        """

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
        """
        Wrapper to set abstract _step to _step_helper passed in __init__.

        Returns:
            None

        Note:
            See `TrainingAbstract._step` for role of this method.
        """
        self._step_helper(self)

    def _build_training_stages(self) -> OrderedDict:
        """
        Wrapper to set abstract _build_training_stages to _build_training_stages_helper passed in __init__.

        Returns:
            An OrderedDict containing the training stages.

        Note:
            See `TrainingAbstract._build_training_stages` for role of this method.
        """
        return self._build_training_stages_helper()

    def _pre_train(self):
        """
        Wrapper to set abstract _pre_train to _pre_train_helper passed in __init__.

        Returns:
            None

        Note:
            See `TrainingAbstract._pre_train` for role of this method.
        """
        self._pre_train_helper(self)

    def _post_train(self):
        """
        Wrapper to set abstract _post_train to _post_train_helper passed in __init__.

        Returns:
            None

        Note:
            See `TrainingAbstract._post_train` for role of this method.
        """
        self._post_train_helper(self)
