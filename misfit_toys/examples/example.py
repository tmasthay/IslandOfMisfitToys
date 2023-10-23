from misfit_toys.fwi.modules.distribution import cleanup, setup
from masthay_helpers.global_helpers import summarize_tensor, DotDict, subdict
from misfit_toys.swiffer import iraise, istr
from masthay_helpers.jupyter import iplot
from misfit_toys.fwi.modules.seismic_data import SeismicProp

from masthay_helpers import peel_final

from abc import ABC, abstractmethod
import os
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import matplotlib.pyplot as plt
from warnings import warn
import copy
import pickle
import numpy as np

from masthay_helpers.global_helpers import dynamic_expand, prettify_dict
from masthay_helpers.jupyter import rules_one, rules_two

from torch.nn.parallel import DistributedDataParallel as DDP


def merge_tensors(*, path, tensor_dict, world_size):
    d = {}
    for k, v in tensor_dict.items():
        d[k] = v([torch.load(f"{path}/{k}_{i}.pt") for i in range(world_size)])
    return d


class ExampleGen:
    def __init__(
        self,
        *,
        prop: SeismicProp,
        training_class: type,
        training_kwargs: dict,
        save_dir: str,
        reduce: dict,
        verbose: int = 1,
        tmp: dict = None,
        **kw,
    ):
        self.prop = prop
        self.save_dir = save_dir
        self.data_save = os.path.abspath(os.path.join(self.save_dir, "data"))
        self.fig_save = os.path.abspath(os.path.join(self.save_dir, "figs"))

        os.makedirs(os.path.join(self.data_save, "tmp"), exist_ok=True)
        os.makedirs(self.fig_save, exist_ok=True)

        self.reduce = reduce
        self.tensor_names = list(reduce.keys())

        self.output_files = {
            e: os.path.join(self.data_save, f"{e}.pt")
            for e in self.tensor_names
        }

        self.training_class = training_class
        self.training_kwargs = training_kwargs

        self.verbose = verbose
        self.tensors = {}
        self.tmp = DotDict(tmp) if tmp is not None else DotDict({})

        self.set_kw(kw)

    def set_kw(self, kw):
        if "output_files" in kw.keys():
            raise ValueError(
                "output_files is generated from tensor_names"
                "and thus should be treated like a reserved keyword"
            )
        self.__dict__.update(kw)

    def _pre_chunk(self, rank, world_size):
        return self.prop

    def _generate_data(self, rank, world_size):
        self.prop = self._pre_chunk(rank, world_size)
        self.prop = self.prop.chunk(rank, world_size)
        self.prop = self.prop.to(rank)
        self.dist_prop = DDP(self.prop, device_ids=[rank])
        self.trainer = self.training_class(
            dist_prop=self.dist_prop,
            rank=rank,
            world_size=world_size,
            **self.training_kwargs,
        )
        self.trainer.train(path=os.path.join(self.data_save, "tmp"))
        self.update_tensors(
            self.trainer.report.dict(), restrict=True, device=rank, detach=True
        )

    def final_result(self, *args, **kw):
        return self._final_result(*args, **kw)

    def _final_result(self, *args, **kw):
        return {
            k: self.plot(**v) for k, v in self._final_dict(*args, **kw).items()
        }

    def _final_dict(self, *args, **kw):
        return self.base_final_dict()

    def base_final_dict(self):
        def rule_builder(*, opts_one, loop_one, opts_two, loop_two):
            return {
                "one": rules_one(opts_info=opts_one, loop_info=loop_one),
                "two": rules_two(opts_info=opts_two, loop_info=loop_two),
            }

        def flatten(data):
            data = [e.reshape(1, -1) for e in data]
            res = torch.stack(data, dim=0)
            second = 2 if res.shape[1] == 1 else 1
            res = res.repeat(1, second, 1)
            return res

        # what does this function do? LOL
        def extend(idx):
            def process(data):
                shape = data[idx].shape
                for i in range(len(data)):
                    if i != idx:
                        data[i] = dynamic_expand(data[i], shape)
                data = torch.stack(data, dim=0)
                return data

            return process

        groups = {
            "Loss": {
                "keys": ["loss"],
                "column_names": ["Frequency", "Epoch"],
                "cols": 1,
                "rules": rule_builder(
                    opts_one={"ylabel": "Loss"},
                    loop_one={"labels": ["Loss"]},
                    opts_two={},
                    loop_two={"labels": ["Loss"]},
                ),
                "data_process": flatten,
            },
            "Obs-Out Filtered": {
                "keys": ["obs_data_filt_record", "out_filt_record"],
                "column_names": [
                    "Shot",
                    "Receiver",
                    "Time Step",
                    "Frequency",
                    "Epoch",
                ],
                "cols": 2,
                "rules": rule_builder(
                    opts_one={"ylabel": "Acoustic Amplitude"},
                    loop_one={"labels": ["Observed Data", "Filtered Output"]},
                    opts_two={},
                    loop_two={"labels": ["Observed Data", "Filtered Output"]},
                ),
                "data_process": None,
            },
            "Out-Out Filtered": {
                "keys": ["out_filt_record", "out_record"],
                "column_names": [
                    "Shot",
                    "Receiver",
                    "Time Step",
                    "Frequency",
                    "Epoch",
                ],
                "cols": 2,
                "rules": rule_builder(
                    opts_one={"ylabel": "Acoustic Amplitude"},
                    loop_one={"labels": ["Filtered Output", "Output"]},
                    opts_two={},
                    loop_two={"labels": ["Filtered Output", "Output"]},
                ),
                "data_process": None,
            },
            "Obs-Out": {
                "keys": ["obs_data", "out_record"],
                "column_names": [
                    "Shot",
                    "Receiver",
                    "Time Step",
                    "Frequency",
                    "Epoch",
                ],
                "cols": 2,
                "rules": rule_builder(
                    opts_one={"ylabel": "Acoustic Amplitude"},
                    loop_one={"labels": ["Observed Data", "Output"]},
                    opts_two={},
                    loop_two={"labels": ["Observed Data", "Output"]},
                ),
                "data_process": extend(1),
            },
            "Velocity": {
                "keys": ["vp_init", "vp_record", "vp_true"],
                "column_names": [
                    "Frequency",
                    "Epoch",
                    "Depth (km)",
                    "Horizontal (km)",
                ],
                "cols": 2,
                "rules": rule_builder(
                    opts_one={"ylabel": "Velocity"},
                    loop_one={
                        "labels": [r"$v_{init}$", r"$v_p$", r"$v_{true}$"]
                    },
                    opts_two={},
                    loop_two={
                        "labels": [r"$v_{init}$", r"$v_p$", r"$v_{true}$"]
                    },
                ),
                "data_process": extend(1),
            },
        }
        return groups

    @staticmethod
    def print_static(*args, level=1, verbose=1, **kw):
        if verbose >= level:
            print(*args, **kw, flush=True)

    def print(self, *args, level=1, **kw):
        Example.print_static(*args, level=level, verbose=self.verbose, **kw)

    def generate_data(self, rank, world_size):
        self.print(f"Running DDP on rank {rank} / {world_size}.", level=2)
        self._generate_data(rank, world_size)
        torch.distributed.barrier()
        if rank == 0:
            self.postprocess(world_size)
        torch.cuda.empty_cache()
        torch.distributed.barrier()

    def postprocess(self, world_size):
        os.makedirs(f"{self.data_save}/tmp", exist_ok=True)
        tmp_path = os.path.join(self.data_save, "tmp")
        st = set(self.tensor_names)
        stk = set(self.tensors.keys())
        unresolved_keys = st - stk
        if not stk.issubset(st):
            raise Example.KeyException(
                self,
                msg=(
                    f"Require self.tensors.keys() to be a subset of"
                    f" self.tensor_names"
                ),
            )
        for k in unresolved_keys:
            self.print("k=", k, level=2)
            curr = []
            for i in range(world_size):
                filename = f"{tmp_path}/{k}_{i}.pt"
                if not os.path.exists(filename):
                    raise Example.KeyException(
                        self, msg=f"File {filename} does not exist. "
                    )
                curr.extend(torch.load(filename))
            if self.reduce is None or self.reduce[k] is None:
                self.tensors[k] = torch.stack(curr)
            else:
                self.tensors[k] = self.reduce[k](curr)

        self.save_all_tensors()

    def add_info(self, *, s, name, **kw):
        return s

    def info_tensor(self, name, **kw):
        s = summarize_tensor(self.tensors[name], heading=name)
        return self.add_info(s=s, name=name, **kw)

    def save_tensor(self, name):
        self.print(f"Saving {name} at {self.output_files[name]}...", end="")
        torch.save(self.tensors[name], self.output_files[name])
        txt_path = self.output_files[name].replace(".pt", "_summary.txt")
        with open(txt_path, "w") as f:
            f.write(self.info_tensor(name))
        self.print(f"SUCCESS", level=1)

    def save_all_tensors(self):
        if set(self.tensors.keys()) != set(self.tensor_names):
            raise Example.KeyException(self)

        for name in self.tensors.keys():
            self.save_tensor(name)

    def load_all_tensors(self):
        self.tensors = {}
        paths_exist = [os.path.exists(f) for f in self.output_files.values()]
        if all(paths_exist):
            for name, f in self.output_files.items():
                self.print(f"Load attempt {name} at {f}...", end="")
                try:
                    self.tensors[name] = torch.load(f)
                except Exception as e:
                    Example.print_static(f"FAIL for {name} at {f}", level=0)
                    raise e
                self.print("SUCCESS")
        else:
            self.print("FAIL")
            self.tensors = None

    def update_tensors(
        self, tensors, *, restrict=False, detach=False, device="cpu"
    ):
        if restrict:
            tensors = {
                k: v for k, v in tensors.items() if k in self.tensor_names
            }
        if detach:
            tensors = {k: v.detach() for k, v in tensors.items()}
        tensors = {k: v.to(device) for k, v in tensors.items()}
        self.tensors.update(tensors)

    def run_rank(self, rank, world_size):
        """
        TODO: generate_data(rank, world_size) really is 'run_rank'. This should all be refactored.
        You should do

        def run():
            self.load_all_tensors()
            if( self.tensors is None ):
                self.tensors = {}
                mp.spawn(self.generate_data, args=(world_size,), nprocs=world_size, join=True)
                self.load_all_tensors()
                if self.tensors is None: raise ValueError(...)
            else:
                self.print('skipping data gen')
            self.plot_data()...
        HOWEVER: maybe there is some reason why you need this with the pickling.
        Don't be overconfidetn...we will keep this structure for now so as to not
        break anything even though it's a bit clumsy from a design standpoint.
        """
        setup(rank, world_size)
        if self.tensors is None:
            self.tensors = {}
            self.generate_data(rank, world_size)
            self.load_all_tensors()
            if self.tensors is None:
                raise ValueError(
                    "FATAL: Data generation failed, check your code"
                )
        else:
            if rank == 0:
                self.print(
                    "Skipping data generation, delete .pt files in "
                    f"{self.data_save} and re-run this script to "
                    "regenerate"
                )
        torch.distributed.barrier()
        # if rank == 0:
        #     self.plot_data()
        #     with open(self.pickle_save, "wb") as f:
        #         self.print(
        #             (
        #                  f"Saving pickle to {self.pickle_save}..."
        #                 f"self.tensors.keys()=={self.tensors.keys()}"
        #             ),
        #             end="",
        #         )
        #         pickle.dump(self, f)
        #         self.print("SUCCESS")
        cleanup()

    def run(self, *args, **kw):
        world_size = torch.cuda.device_count()
        if world_size == 0:
            raise ValueError(
                "\nFATAL: No GPUs detected, check your system.\n"
                "    We currently do not support CPU-only training."
            )
        # self.losses = torch.empty(world_size + 1)
        self.load_all_tensors()
        mp.spawn(
            self.run_rank, args=(world_size,), nprocs=world_size, join=True
        )
        self.load_all_tensors()
        return self.final_result(*args, **kw)
        # with open(self.pickle_save, "rb") as f:
        #     self.print(f"Loading pickle from {self.pickle_save}...", end="")
        #     self = pickle.load(f)
        #     self.print(
        #         f"SUCCESS...deleting self.pickle_save...={self.pickle_save}",
        #         end="",
        #     )
        #     os.remove(self.pickle_save)
        #     self.print("SUCCESS")
        # return self

    def plot(self, *, keys, column_names, cols, rules, data_process=None):
        data = [self.tensors[k].detach().cpu() for k in keys]
        if data_process is not None:
            data = data_process(data)
        else:
            data = torch.stack(data, dim=0)
        return iplot(
            data=data, column_names=column_names, cols=cols, rules=rules
        )

    @staticmethod
    def first_elem(x):
        return x[0]

    @staticmethod
    def mean_reduce(x):
        return torch.stack(x).mean(dim=0)

    class KeyException(Exception):
        def __init__(self, ex, msg=None):
            super().__init__(Example.KeyException.build_base_msg(ex, msg))

        @staticmethod
        def build_base_msg(ex, msg):
            name_minus_keys = set(ex.tensor_names) - set(ex.tensors.keys())
            keys_minus_name = set(ex.tensors.keys()) - set(ex.tensor_names)
            msg = "" if msg is None else msg + "\n"
            s = (
                f"FATAL: self.tensors() != self.tensor_names\n{msg}",
                istr(
                    "Debug info below:\n",
                    f"self.tensor_names={ex.tensor_names}",
                    f"self.tensors.keys()={ex.tensors.keys()}",
                    f"keys() - tensor_names={keys_minus_name}",
                    f"tensor_names - keys()={name_minus_keys}",
                ),
                istr(
                    "USER RESPONSIBILITY\n",
                    "Any unresolved tensor names in self.tensor_names",
                    "need to be set in one of two ways in abstract ",
                    "self._generate_data method.",
                    istr(
                        "\n",
                        "(1) Explicitly set (params synced by DDP)\n",
                        (
                            "(2) Implicitly set by saving to"
                            ' f"{ex.data_save}/{key}_{rank}.pt" for'
                            " each rank (unsynced metadata, e.g. loss"
                            " history)"
                        ),
                    ),
                ),
            )
            return s


class Example(ExampleGen):
    def __init__(
        self,
        *,
        prop_kwargs,
        training_class: type,
        training_kwargs: dict,
        save_dir: str,
        reduce: dict,
        verbose: int = 1,
        tmp: dict = None,
        **kw,
    ):
        super().__init__(
            prop=SeismicProp(**prop_kwargs),
            training_class=training_class,
            training_kwargs=training_kwargs,
            save_dir=save_dir,
            reduce=reduce,
            verbose=verbose,
            tmp=tmp,
            **kw,
        )


class ExampleComparator:
    def __init__(
        self,
        *examples,
        data_save="compare/data",
        fig_save="compare/figs",
        protect=None,
        log=0,
    ):
        if len(examples) != 2:
            raise ValueError(
                "FATAL: ExampleComparator requires exactly 2 examples"
            )
        self.first = examples[0].run()
        self.second = examples[1].run()

        if set(self.first.tensor_names) != set(self.second.tensor_names):
            raise ValueError(
                "FATAL: tensor_names for both examples must match, got\n"
                f"    (1) {self.first.tensor_names}\n"
                f"    (2) {self.second.tensor_names}"
            )
        if protect is None:
            self.protect = []
        else:
            self.protect = protect

        self.data_save = data_save
        self.fig_save = fig_save
        self.log = log

        self.dummy_first_path()

        os.makedirs(self.data_save, exist_ok=True)
        os.makedirs(self.fig_save, exist_ok=True)

    def dummy_first_path(self):
        self.first.old_data_save = self.first.data_save
        self.first.old_fig_save = self.first.fig_save
        self.first.old_output_files = self.first.output_files
        self.first.data_save = self.data_save
        self.first.fig_save = self.fig_save
        self.first.output_files = {
            e: os.path.join(self.data_save, f"{e}.pt")
            for e in self.first.tensor_names
        }

    def compare(self, **kw):
        if (
            set(self.first.tensors.keys()) != set(self.second.tensors.keys())
            or set(self.first.tensors.keys()) != set(self.first.tensor_names)
            or set(self.second.tensors.keys()) != set(self.second.tensor_names)
        ):
            raise ValueError(
                "\n\n\nFATAL: keys for both examples must match each other and"
                " their respective tensor_names attribute.\nNOTE:"
                " Example.self.tensors is initialized to an empty dict.\n   "
                " It is the responsibility of the user to populate it, usually"
                " within _generate_data concretization.\n    It is expected"
                " that by the time _generate_data is finished that"
                " self.tensors.keys() == self.tensor_names. Debugging info"
                " below\n    (1) self.first.tensors.keys():"
                f" {self.first.tensors.keys()}\n    (2)"
                f" self.second.tensors.keys(): {self.second.tensors.keys()}\n  "
                f"  (3) self.first.tensor_names: {self.first.tensor_names}\n   "
                f" (4) self.second.tensor_names: {self.second.tensor_names}\n"
            )

        self.first.data_save = self.data_save
        self.first.fig_save = self.fig_save
        for name in self.first.tensor_names:
            if name not in self.protect:
                self.first.tensors[name] = (
                    self.first.tensors[name] - self.second.tensors[name]
                )
                self.first.tensors[name] = self.first.tensors[name].abs()
                if self.log == 1:
                    self.first.tensors[name] = torch.log(
                        self.first.tensors[name]
                    )
                elif self.log == 2:
                    self.first.tensors[name] = torch.log(
                        1.0 + self.first.tensors[name]
                    )
        self.first.save_all_tensors()
        self.first.plot_data(**kw)
