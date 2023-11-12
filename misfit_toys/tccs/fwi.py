import os

import torch
import torch.multiprocessing as mp
from masthay_helpers.global_helpers import (
    DotDict,
    bstr,
    dynamic_expand,
    summarize_tensor,
)
from masthay_helpers.jupyter import iplot, rules_one, rules_two
from torch.nn.parallel import DistributedDataParallel as DDP

from misfit_toys.fwi.modules.distribution import cleanup, setup
from misfit_toys.fwi.modules.seismic_data import SeismicProp
from misfit_toys.swiffer import istr
from misfit_toys.utils import parse_path


def merge_tensors(*, path, tensor_dict, world_size):
    d = {}
    for k, v in tensor_dict.items():
        d[k] = v([torch.load(f"{path}/{k}_{i}.pt") for i in range(world_size)])
    return d


class FWI:
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
        disk_device='cpu',
        **kw,
    ):
        self.prop = prop
        self.save_dir = parse_path(save_dir)
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

        self.disk_device = disk_device

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
        res = self._final_dict(*args, **kw)
        items = res.items()
        d = {}
        for k, v in items:
            d[k] = self.plot(**v)
        return d

    def _final_dict(self, *args, **kw):
        return self.base_final_dict(**kw)

    def base_final_dict(self, **kw):
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
        #     pretty sure that it makes the data conform to the shape of the
        #     specified index (idx).
        #     Input data is a list of torch tensors.
        def extend(idx):
            def process(data):
                shape = data[idx].shape
                for i in range(len(data)):
                    if i != idx:
                        data[i] = dynamic_expand(data[i], shape)
                data = torch.stack(data, dim=0)
                return data

            return process

        all_keys = [
            'Loss',
            'Obs-Out Filtered',
            'Out-Out Filtered',
            'Obs-Out',
            'Velocity',
        ]
        all_subkeys = ['keys', 'column_names', 'cols', 'rules', 'data_process']
        for k in all_keys:
            kw[k] = kw.get(k, {})
            kw[k]['keys'] = kw[k].get('keys', [])
            kw[k]['column_names'] = kw[k].get('column_names', [])
            kw[k]['cols'] = kw[k].get('cols', None)
            kw[k]['rules'] = kw[k].get(
                'rules',
                {
                    'opts_one': {},
                    'loop_one': {},
                    'opts_two': {},
                    'loop_two': {},
                },
            )
            kw[k]['data_process'] = kw[k].get('data_process', None)

        def extend_group(
            *, keys, column_names, cols, rules, data_process, name
        ):
            additions = kw.get(name, {})
            keys.extend(additions.get('keys', []))
            column_names.extend(additions.get('column_names', []))
            cols = (
                cols
                if additions.get('cols', None) is None
                else additions['cols']
            )
            rules = {
                'opts_one': {
                    **rules['opts_one'],
                    **additions.get('opts_one', {}),
                },
                'loop_one': {
                    **rules['loop_one'],
                    **additions.get('loop_one', {}),
                },
                'opts_two': {
                    **rules['opts_two'],
                    **additions.get('opts_two', {}),
                },
                'loop_two': {
                    **rules['loop_two'],
                    **additions.get('loop_two', {}),
                },
            }
            data_process = (
                data_process
                if additions.get('data_process', None) is None
                else additions['data_process']
            )
            return {
                'keys': keys,
                'column_names': column_names,
                'cols': cols,
                'rules': rule_builder(**rules),
                'data_process': data_process,
            }

        groups = {
            "Loss": {
                "keys": ["loss"],
                "column_names": ["Frequency", "Epoch"],
                "cols": 1,
                "rules": dict(
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
                "rules": dict(
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
                "rules": dict(
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
                "rules": dict(
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
                "rules": dict(
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
        for k, v in groups.items():
            groups[k] = extend_group(**v, name=k)

        # raise ValueError('debug')
        return groups

    @staticmethod
    def print_static(*args, level=1, verbose=1, **kw):
        if verbose >= level:
            print(*args, **kw, flush=True)

    def print(self, *args, level=1, **kw):
        FWI.print_static(*args, level=level, verbose=self.verbose, **kw)

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
            e = FWI.KeyException(
                self,
                msg=(
                    f"Require self.tensors.keys() to be a subset of"
                    f" self.tensor_names"
                ),
            )
            print(e, flush=True)
            raise e
        for k in unresolved_keys:
            self.print("k=", k, level=2)
            curr = []
            for i in range(world_size):
                filename = f"{tmp_path}/{k}_{i}.pt"
                if not os.path.exists(filename):
                    raise FWI.KeyException(
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
            raise FWI.KeyException(self)

        for name in self.tensors.keys():
            if self.disk_device is not None:
                self.tensors[name] = self.tensors[name].to(self.disk_device)
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
                    FWI.print_static(f"FAIL for {name} at {f}", level=0)
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
        cleanup()

    def run(self, *args, **kw):
        world_size = torch.cuda.device_count()
        if world_size == 0:
            raise ValueError(
                "\nFATAL: No GPUs detected, check your system.\n"
                "    We currently do not support CPU-only training."
            )
        self.load_all_tensors()
        mp.spawn(
            self.run_rank, args=(world_size,), nprocs=world_size, join=True
        )
        self.load_all_tensors()
        return self.final_result(*args, **kw)

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
            super().__init__(FWI.KeyException.build_base_msg(ex, msg))
            print(self, flush=True)

        @staticmethod
        def build_base_msg(ex, msg):
            name_minus_keys = set(ex.tensor_names) - set(ex.tensors.keys())
            keys_minus_name = set(ex.tensors.keys()) - set(ex.tensor_names)
            msg = "" if msg is None else msg + "\n"
            s = bstr(
                f"FATAL: self.tensors() != self.tensor_names\n{msg}",
                istr(
                    "Debug info below:\n",
                    f"self.tensor_names={ex.tensor_names}",
                    f"self.tensors.keys()={ex.tensors.keys()}",
                    f"keys() - tensor_names={keys_minus_name}",
                    f"tensor_names - keys()={name_minus_keys}\n",
                ),
                istr(
                    "USER RESPONSIBILITY\n",
                    "Any unresolved tensor names in self.tensor_names",
                    "need to be set in one of two ways in abstract ",
                    "self._generate_data method.",
                    istr(
                        "\n",
                        "(1) Explicitly set (params synced by DDP)\n",
                        "(2) Implicitly set by saving to"
                        " f\"{ex.data_save}/{key}_{rank}.pt\" for"
                        " each rank (unsynced metadata, e.g. loss"
                        " history)",
                    ),
                ),
            )
            return s


class FWIPass(FWI):
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
