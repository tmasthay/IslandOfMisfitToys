from abc import ABC, abstractmethod
from masthay_helpers.global_helpers import get_print, DotDict
import torch
from tabulate import tabulate as tab
from collections import OrderedDict
from misfit_toys.tccs.modules.seismic_data import SeismicProp
from typing import Tuple, List
from rich.table import Table
from rich.console import Console


# TrainingAllAbstract
# This is the true base abstract class with three abstract methods
#     _step -- the main training logic for a single step
#     _pre_train -- any computation before training has started
#     _post_train -- any computation after training has occurred
class TrainingAbstract(ABC):
    """Abstract base class for training

    Attributes:
        dist_prop: Distribution object
        rank: Rank of the current process
        world_size: Total number of processes
        verbose: Verbosity level
        optimizer: Optimizer object
        scheduler: Scheduler object
        loss: Loss function
        training_stages: OrderedDict of training stages
        custom: Custom attributes
        report: Report attributes
        trainable_records: Trainable records

    Abstract Methods:
        _step: Main training logic for a single step
        _pre_train: Any computation before training has started
        _post_train: Any computation after training has occurred

    Public Methods:
        train: Main training loop
        step: Single training step

    Protected Methods:
        _report_info: Report info
        _train: Train
        _pre_epoch: Pre-epoch
        _post_epoch: Post-epoch

    Private Methods:
        __build_scheduler: Build scheduler
        __setup_trainable_records: Setup trainable records
        __update_step_info: Update step info
        __reset_optimizer: Reset optimizer
        __recursive_train: Recursive training

    """

    def __init__(
        self,
        *,
        dist_prop: SeismicProp,
        rank: int,
        world_size: int,
        verbose: int = 1,
        optimizer: Tuple[torch.optim.Optimizer, dict],
        scheduler: List[Tuple[torch.optim.lr_scheduler._LRScheduler, dict]],
        loss: torch.nn.Module,
        training_stages: OrderedDict = None,
        **kw,
    ):
        """Constructor

        Args:
            dist_prop (SeismicProp): _description_
            rank (int): _description_
            world_size (int): _description_
            optimizer (Tuple[torch.optim.Optimizer, dict]): _description_
            scheduler (_type_): _description_
            verbose (int, optional): _description_. Defaults to 1.
            training_stages (OrderedDict, optional): _description_. Defaults to None.
        """
        self.dist_prop = dist_prop
        self.rank = rank
        self.world_size = world_size
        self.optimizer = optimizer[0](
            self.dist_prop.module.parameters(), **optimizer[1]
        )
        self.scheduler = self.__build_scheduler(scheduler)
        self.loss = loss
        self.print, self.printj = get_print(_verbose=verbose)
        self.custom = DotDict(kw)
        self.report = DotDict({'loss_record': []})
        if type(training_stages) is int:
            self.training_stages = OrderedDict(
                (
                    'Epochs',
                    {
                        'data': range(training_stages),
                        'preprocess': lambda x: None,
                        'postprocess': lambda x: None,
                    },
                )
            )
        else:
            self.training_stages = training_stages
        self.__setup_trainable_records()
        self.__setup_report_table()

    def _step(self):
        """Abstract step method. Must be implemented by subclass."""

    @abstractmethod
    def _pre_train(self):
        """Abstract pre-training method. Must be implemented by subclass."""

    @abstractmethod
    def _post_train(self):
        """Abstract post-training method. Must be implemented by subclass."""

    def train(self):
        self._pre_train()
        self._train()
        self._post_train()

    def step(self):
        loss = 0.0
        calls = 0

        def closure():
            nonlocal loss, calls
            calls += 1
            self.optimizer.zero_grad()
            loss_local, other_info = self._step()
            if calls == 1:
                loss = loss_local
                self.__update_step_info(loss_local, other_info)
            loss_local.backward()
            return loss_local

        self.optimizer.step(closure)
        if self.scheduler is not None:
            self.scheduler.step()
        self._report_info(calls=calls)
        return loss

    def reset_optimizer(self):
        opt_type = type(self.optimizer)
        self.optimizer = opt_type(
            self.dist_prop.module.parameters(), **self.optimizer.defaults
        )

    def _report_info(self, *, calls, **kw):
        data = [self.epoch, f'{self.report.loss[-1]:.4e}', self.rank, calls]
        if self.epoch == 0:
            header = ['Epoch', 'Loss', 'Rank', '# of Gradient Propagations']
        the_table = tab([data], headers=header, tablefmt='plain')
        self.print(the_table)

    def _train(self):
        for epoch in range(self.n_epochs):
            self.epoch = epoch
            self._pre_epoch()
            self.step()
            self._post_epoch()

    def _report_info(self, *, calls, **kw):
        self.custom.report_table.add_row(
            str(len(self.report.loss_record)),
            f'{self.report.loss_record[-1]:.8f}',
            str(self.rank),
            str(calls),
        )
        self.custom.console.print('\033[H\033[J', end='\n')
        self.custom.console.print(self.custom.report_table)
        # the_table = tab([data], tablefmt='plain')
        # self.print(the_table)

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
            level_info['data'],
            level_info['preprocess'],
            level_info['postprocess'],
        )

        idt = '    ' * depth
        for item in data:
            self.print(f"{idt}Preprocessing {level_name} {item}", verbose=2)
            preprocess(self, item)  # Assuming preprocess takes 'self'

            self.__recursive_train(
                level_data=level_data, depth=depth + 1, max_depth=max_depth
            )

            self.print(f"{idt}Postprocessing {level_name} {item}", verbose=2)
            postprocess(self, item)

    def __update_step_info(self, loss_local, other_info):
        self.report.loss_record.append(loss_local.detach().cpu())

        if not other_info:
            return

        for k, v in other_info.items():
            if k not in self.report.keys():
                self.report.set(k, [])
            self.report.get(k).append(v.detach().cpu())
        self.print(self.report.str(), verbose=2)

    def __build_scheduler(self, scheduler):
        if scheduler is None:
            return None
        schedule_list = []
        for s in scheduler:
            schedule_list.append(s[0](self.optimizer, **s[1]))
        scheduler = torch.optim.lr_scheduler.ChainedScheduler(schedule_list)
        return scheduler

    def __setup_trainable_records(self):
        self.trainable_records = DotDict({})
        for name, p in self.dist_prop.module.named_parameters():
            if p.requires_grad:
                key = name.replace('.p', '')
                self.report.set(key + '_record', [])

    def __setup_report_table(self):
        self.custom.set('report_table', Table())
        self.custom.report_table.add_column('Epoch')
        self.custom.report_table.add_column('Loss')
        self.custom.report_table.add_column('Rank')
        self.custom.report_table.add_column('Optimizer Calls')
        self.custom.setdefault('tmp_out_base', '/tmp/tmp')
        self.custom.set(
            'console',
            Console(
                file=open(f'{self.custom.tmp_out_base}{self.rank}.txt', 'w')
            ),
        )
        self.custom.console.print(self.custom.report_table)


# TrainingAbstractStep
# Abstract class with _pre_train and _post_train defaults implemented
class Training(TrainingAbstract):
    def __init__(
        self,
        *,
        step,
        pre_train=None,
        post_train=None,
        dist_prop,
        rank,
        world_size,
        verbose=1,
        optimizer,
        scheduler,
        loss,
        training_stages: OrderedDict = None,
        **kw,
    ):
        super().__init__(
            dist_prop=dist_prop,
            rank=rank,
            world_size=world_size,
            verbose=verbose,
            optimizer=optimizer,
            scheduler=scheduler,
            loss=loss,
            training_stages=training_stages,
            **kw,
        )
        self.step_dummy = step
        self.pre_train_dummy = pre_train
        self.post_train_dummy = post_train

    def __no_impl_warning(self, name):
        self.print(
            f'WARNING: {name} not overridden in subclass'
            'To silence this warning, set verbosity <= 1',
            verbose=2,
        )

    def _pre_train(self):
        if self.pre_train_dummy is None:
            self.__no_impl_warning('_pre_train')
        else:
            self.pre_train_dummy(self)

    def _post_train(self):
        if self.post_train_dummy is None:
            self.__no_impl_warning('_post_train')
        else:
            self.post_train_dummy(self)

    def _step(self):
        return self.step_dummy(self)
