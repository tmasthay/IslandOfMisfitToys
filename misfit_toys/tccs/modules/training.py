from abc import ABC, abstractmethod
from masthay_helpers.global_helpers import get_print, DotDict
import torch
from tabulate import tabulate as tab


class TrainingFullyAbstract(ABC):
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
        n_epochs,
        **kw,
    ):
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
        self.report = DotDict({'loss': []})
        self.n_epochs = n_epochs
        self.__setup_trainable_records()

    @abstractmethod
    def _step(self):
        """Abstract step method. Must be implemented by subclass."""

    @abstractmethod
    def _pre_train(self):
        """Abstract pre-training method. Must be implemented by subclass."""

    @abstractmethod
    def _post_train(self):
        """Abstract post-training method. Must be implemented by subclass."""

    @abstractmethod
    def _pre_epoch(self):
        """Abstract pre-epoch method. Must be implemented by subclass."""

    @abstractmethod
    def _post_epoch(self):
        """Abstract post-epoch method. Must be implemented by subclass."""

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
        self.report_info(calls=calls)
        return loss

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

    def __reset_optimizer(self):
        opt_type = type(self.optimizer)
        self.optimizer = opt_type(
            self.dist_prop.module.parameters(), **self.optimizer.defaults
        )

    def __update_step_info(self, loss_local, other_info):
        self.report.loss.append(loss_local.detach())

        if not other_info:
            return

        for k, v in other_info.items():
            if k not in self.report.keys():
                self.report.set(k, [])
            self.report.get(k).append(v.detach())

    def __build_scheduler(self, scheduler):
        if scheduler is None:
            return None
        schedule_list = []
        for s in scheduler:
            schedule_list.append(s[0](self.optimizer, **s[1]))
        scheduler = torch.optim.lr_scheduler.ChainedScheduler(schedule_list)
        return scheduler

    def __setup_trainable_records(self):
        self.trainable_records = DotDict()
        for name, p in self.dist_prop.module.named_parameters():
            if p.requires_grad:
                key = name.replace('.p', '')
                self.report.set(
                    key + '_record', torch.zeros(self.n_epochs, *p.shape)
                )


class Training(TrainingFullyAbstract):
    def __no_impl_warning(self, name):
        self.print(
            f'WARNING: {name} not overridden in subclass'
            'To silence this warning, set verbosity <= 1',
            2,
        )

    def _pre_train(self):
        self.__no_impl_warning('_pre_train')

    def _post_train(self):
        self.__no_impl_warning('_post_train')

    def _pre_epoch(self):
        self.__no_impl_warning('_pre_epoch')

    def _post_epoch(self):
        self.__no_impl_warning('_post_epoch')


class TrainingMultiscale(Training):
    def _step(self):
        pass

    def _pre_train(self):
        pass

    def _post_train(self):
        pass

    def _pre_epoch(self):
        pass

    def _post_epoch(self):
        pass
