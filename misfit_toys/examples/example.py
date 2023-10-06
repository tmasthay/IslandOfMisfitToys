from misfit_toys.fwi.modules.distribution import cleanup, setup
from misfit_toys.utils import summarize_tensor
from misfit_toys.swiffer import iraise, istr

from abc import ABC, abstractmethod
import os
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import matplotlib.pyplot as plt
from warnings import warn
import copy
import pickle


class Example(ABC):
    def __init__(
        self,
        *,
        data_save,
        fig_save,
        pickle_save=None,
        tensor_names,
        verbose=1,
        subplot_args={},
        plot_args={},
        **kw,
    ):
        self.data_save = os.path.abspath(data_save)
        self.fig_save = os.path.abspath(fig_save)
        self.pickle_save = pickle_save

        os.makedirs(f"{self.data_save}/tmp", exist_ok=True)
        os.makedirs(self.fig_save, exist_ok=True)

        self.tensor_names = tensor_names
        self.output_files = {
            e: os.path.join(data_save, f"{e}.pt") for e in self.tensor_names
        }
        self.verbose = verbose
        self.tensors = {}
        self.subplot_args = subplot_args
        self.plot_args = plot_args
        if "output_files" in kw.keys():
            raise ValueError(
                "output_files is generated from tensor_names"
                "and thus should be treated like a reserved keyword"
            )
        self.__dict__.update(kw)

        if self.pickle_save is None:
            random_int = torch.randint(0, 1000000, (1,)).item()
            curr_prop = f"/tmp/pickle_{random_int}.pkl"
            num_tries = 10
            curr_try = 0
            while os.path.exists(curr_prop) and curr_try < num_tries:
                curr_prop = f"/tmp/pickle_{random_int}.pkl"
                curr_try += 1
                if curr_try == num_tries:
                    raise ValueError(
                        "FATAL: Tried to instantiate Example with pickle "
                        f"path {curr_prop} with no success after {num_tries} "
                        "tries. Clean up your /tmp directory and try again."
                    )
            self.pickle_save = curr_prop

        self.debug_save = f"{self.data_save}/debug"

    @abstractmethod
    def _generate_data(self, rank, world_size):
        pass

    @abstractmethod
    def plot_data(self, **kw):
        pass

    @staticmethod
    def print_static(*args, level=1, verbose=1, **kw):
        if verbose >= level:
            print(*args, **kw, flush=True)

    @staticmethod
    def plot_inv_record(
        *,
        fig_save,
        init,
        true,
        record,
        name,
        labels,
        subplot_args=None,
        plot_args=None,
        verbose=1,
    ):
        default_subplot = dict(figsize=(10.5, 10.5), sharex=True, sharey=True)
        default_plot_keys = dict(cmap="gray", aspect="auto")

        if subplot_args is None:
            subplot_args = {}

        if plot_args is None:
            plot_args = {}

        subplot_args = {**default_subplot, **subplot_args}
        plot_args = {**default_plot_keys, **plot_args}

        # Extract the shape of the tensor's dimensions prior to the last 2 dimensions
        lengths = record.shape[:-2]

        # Reshape the record tensor
        reshaped_record = record.view(-1, *record.shape[-2:])

        # Create the index map
        indices = [torch.arange(length) for length in lengths]
        grid = torch.meshgrid(*indices)
        index_map = torch.stack(grid, dim=-1).view(-1, len(lengths))

        def subtitle(i):
            curr = [
                f"{labels[j][0]}={labels[j][1][index_map[i, j]]}"
                for j in range(len(lengths))
            ]
            return "(" + ", ".join(curr) + ")"

        if plot_args.get("transpose", False):
            init = init.T
            true = true.T
            reshaped_record = reshaped_record.transpose(1, 2)
            del plot_args["transpose"]

        fig, ax = plt.subplots(3, **subplot_args)
        Example.print_static(
            f"Plotting final results {name} at {fig_save}/final_{name}.jpg...",
            end="",
            verbose=verbose,
        )
        ax[0].imshow(init, **plot_args)
        ax[0].set_title(f"{name} Initial")
        ax[2].imshow(true, **plot_args)
        ax[2].set_title(f"{name} Ground Truth")
        Example.print_static("SUCCESS", verbose=verbose)
        for i, curr in enumerate(reshaped_record):
            save_file_name = f"{fig_save}/{name}_{i}.jpg"
            Example.print_static(
                f"Plotting {save_file_name} with params={subtitle(i)}...",
                end="",
            )
            curr_ax1 = ax[1].imshow(curr, **plot_args)
            ax[1].set_title(f"{name} {subtitle(i)}")
            fig.colorbar(
                curr_ax1, ax=ax.ravel().tolist(), orientation="vertical"
            )
            plt.tight_layout()
            plt.savefig(save_file_name)
            ax[1].cla()
            if len(fig.axes) > 3:
                fig.delaxes(fig.axes[-1])
            Example.print_static("SUCCESS", verbose=verbose)
        plt.close()
        Example.print_static(
            f"gif creation attempt for {name} at {fig_save}/{name}.gif...",
            end="",
        )
        os.system(
            "convert -delay 100 -loop 0 "
            f"$(ls -tr {fig_save}/{name}_*.jpg) "
            f"{fig_save}/{name}.gif"
        )
        Example.print_static("SUCCESS", verbose=verbose)
        Example.print_static(
            f"cleaning up {fig_save}/{name}_*.jpg...", end="", verbose=verbose
        )
        os.system(f"rm $(ls -t {fig_save}/{name}_*.jpg | tail -n +2)")
        os.system(
            f"mv $(ls {fig_save}/{name}_[0-9]*.jpg) {fig_save}/{name}_final.jpg"
        )
        Example.print_static("SUCCESS", verbose=verbose)

    def plot_inv_record_auto(
        self, *, name, labels, subplot_args={}, plot_args={}
    ):
        subplot_args = {**self.subplot_args, **subplot_args}
        plot_args = {**self.plot_args, **plot_args}
        Example.plot_inv_record(
            fig_save=self.fig_save,
            init=self.tensors[f"{name}_init"],
            true=self.tensors[f"{name}_true"],
            record=self.tensors[f"{name}_record"],
            name=name,
            labels=labels,
            subplot_args=subplot_args,
            plot_args=plot_args,
            verbose=self.verbose,
        )

    def generate_data(self, rank, world_size):
        self.print(f"Running DDP on rank {rank} / {world_size}.", level=2)
        self._generate_data(rank, world_size)
        torch.distributed.barrier()
        if rank == 0:
            self.postprocess(world_size)
            self.save_all_tensors()
        torch.distributed.barrier()

    def postprocess(self, world_size, reduce=None):
        os.makedirs(f"{self.data_save}/tmp", exist_ok=True)
        tmp_path = os.path.join(self.data_save, "tmp")
        st = set(self.tensor_names)
        stk = set(self.tensors.keys())
        unresolved_keys = st - stk
        assert not (stk - st), f"tensors.keys() - tensor_names = {stk - st}"
        for k in unresolved_keys:
            self.print("k=", k)
            curr = []
            for i in range(world_size):
                filename = f"{tmp_path}/{k}_{i}.pt"
                if not os.path.exists(filename):
                    iraise(
                        ValueError,
                        f"FATAL: Could not find {filename} in postprocess.\n",
                        istr(
                            "Debug info below:\n",
                            f"self.tensor_names={self.tensor_names}",
                            f"self.tensors.keys()={self.tensors.keys()}",
                            f"unresolved_keys={unresolved_keys}",
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
                                    ' f"{self.data_save}/{key}_{rank}.pt" for'
                                    " each rank (unsynced metadata, e.g. loss"
                                    " history)"
                                ),
                            ),
                        ),
                    )
                curr.append(torch.load(filename))
            if reduce is None or reduce[k] is None:
                self.tensors[k] = curr[0]
            elif reduce[k] == "stack":
                self.tensors[k] = torch.stack(curr)
            elif reduce[k] == "sum":
                self.tensors[k] = torch.stack(curr).sum(dim=0)
            elif reduce[k] == "mean":
                self.tensors[k] = torch.stack(curr).mean(dim=0)
            else:
                self.tensors[k] = reduce[k](curr)

        stk = set(self.tensors.keys())
        assert st == stk, istr(
            f"FATAL",
            f"self.tensor_names={self.tensor_names}\n",
            f"self.tensors.keys()={self.tensors.keys()}\n",
            f"This assertion should never occur!\n",
            "Please report this bug to the IslandOfMisfitToys developers!\n",
            "Debug info below\n",
            f"tensor_names={st}\n",
            f"tensors.keys()={stk}\n",
            f"tensor_names - tensors.keys()={st - stk}\n",
            f"tensors.keys() - tensor_names={stk - st}\n",
        )

    def plot_field(
        self,
        *,
        field,
        title=None,
        aspect="auto",
        cmap="seismic",
        cbar="uniform",
        transpose=False,
        **kw,
    ):
        self.plot_field_default(
            field=field,
            title=title,
            aspect=aspect,
            cmap=cmap,
            cbar=cbar,
            transpose=transpose,
            **kw,
        )

    @staticmethod
    def static_plot_field_default(
        *,
        field,
        tensor,
        fig_save,
        title=None,
        aspect="auto",
        cmap="seismic",
        cbar="uniform",
        transpose=False,
        **kw,
    ):
        u = tensor.detach().cpu()
        title = field if title is None else title
        if len(u.shape) not in [2, 3]:
            raise NotImplementedError(
                "Only support 2,3 dimensional tensors, got shape={u.shape}"
            )
        if len(u.shape) == 2:
            u = u.unsqueeze(0)

        if transpose:
            u = torch.transpose(u, 1, 2)

        vmin = u.min()
        vmax = u.max()
        full_kw = kw
        full_kw.update(dict(vmin=vmin, vmax=vmax, cmap=cmap, aspect=aspect))
        for i, curr in enumerate(u):
            plt.clf()
            if cbar == "variable" or cbar == "dynamic":
                vmin, vmax = curr.min(), curr.max()
                full_kw.update(dict(vmin=vmin, vmax=vmax))
            plt.imshow(curr, **full_kw)
            plt.title(f"{title} {i}")
            if cbar:
                plt.colorbar()
            plt.savefig(f"{fig_save}/{field}_{i}.jpg")
            plt.clf()
        os.system(
            "convert -delay 100 -loop 0 "
            f"$(ls -tr {fig_save}/{field}_*.jpg) "
            f"{fig_save}/{field}.gif"
        )
        os.system(f"rm {fig_save}/{field}_*.jpg")

    def plot_field_default(
        self,
        *,
        field,
        title=None,
        aspect="auto",
        cmap="seismic",
        cbar="uniform",
        **kw,
    ):
        Example.static_plot_field_default(
            field=field,
            tensor=self.tensors[field],
            fig_save=self.fig_save,
            title=title,
            aspect=aspect,
            cmap=cmap,
            cbar=cbar,
            **kw,
        )

    def print(self, *args, level=1, **kw):
        Example.print_static(*args, level=level, verbose=self.verbose, **kw)

    def add_info(self, *, s, name, **kw):
        return s

    def info_tensor(self, name, **kw):
        s = summarize_tensor(self.tensors[name], heading=name)
        return self.add_info(s=s, name=name, **kw)

    def save_tensor(self, name):
        self.print(f"Saving {name} at {self.output_files[name]}...", end="")
        torch.save(self.tensors[name], self.output_files[name])
        txt_path = self.output_files[name].replace(".pt", ".tensor_summary")
        with open(txt_path, "w") as f:
            f.write(self.info_tensor(name))
        self.print(f"SUCCESS", level=1)

    def save_all_tensors(self):
        if set(self.tensors.keys()) != set(self.tensor_names):
            in_key_not_in_names = set(self.tensors.keys()) - set(
                self.tensor_names
            )
            in_names_not_in_key = set(self.tensor_names) - set(
                self.tensors.keys()
            )
            raise ValueError(
                "\nFATAL: tensor_names and self.tensors.keys() do not match.\n"
                "    It is the responsibility of the user to ensure that by\n"
                "        the end of calls to your overload of the abstract\n"
                "        `_generate_data function` that all tensor_names keys\n"
                "        are set in Example.self.tensors.\n"
                f'    Debugging info below{""}:\n'
                f"    tensor_names: {set(self.tensor_names)}\n"
                f"    self.tensors.keys(): {set(self.tensors.keys())}\n"
                f"    in_key_not_in_names: {in_key_not_in_names}\n"
                f"    in_names_not_in_key: {in_names_not_in_key}"
            )

        for name in self.tensors.keys():
            self.save_tensor(name)

    def load_all_tensors(self):
        assert all(
            [
                a == b
                for a, b in zip(self.tensor_names, self.output_files.keys())
            ]
        )
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

    def plot_loss(self, **kw):
        style = ["-", "--", "-.", ":"]
        color = ["b", "g", "r", "c", "m", "y", "k"]
        self.tensors["loss"] = self.tensors["loss"].to("cpu")
        self.print(f'loss inside plot_loss: {self.tensors["loss"]}')
        world_size = self.tensors["loss"].shape[0]
        reshaped_loss = self.tensors["loss"].view(world_size, -1)
        full_loss = reshaped_loss.sum(dim=0)
        plt.clf()
        for r in range(world_size):
            label = f"rank {r}"
            print(f"label={label}")
            plt.plot(
                reshaped_loss[r],
                label=label,
                linestyle=style[r % len(style)],
                color=color[r % len(color)],
            )
        plt.plot(
            full_loss,
            label="Full loss",
            linestyle=style[world_size % len(style)],
            color=color[world_size % len(color)],
        )
        plt.title("Loss")
        plt.legend()
        plt.savefig(f"{self.fig_save}/loss.jpg")
        plt.clf()

    def plot_fields(self, **kw):
        for fld, tnsr in self.tensors.items():
            self.plot_field(field=fld, tensor=tnsr, **kw)
        self.plot_loss(**kw)

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
        self.load_all_tensors()
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
        if rank == 0:
            self.plot_data()
            with open(self.pickle_save, "wb") as f:
                self.print(
                    (
                        f"Saving pickle to {self.pickle_save}..."
                        f"self.tensors.keys()=={self.tensors.keys()}"
                    ),
                    end="",
                )
                pickle.dump(self, f)
                self.print("SUCCESS")
        cleanup()

    def run(self):
        world_size = torch.cuda.device_count()
        if world_size == 0:
            raise ValueError(
                "\nFATAL: No GPUs detected, check your system.\n"
                "    We currently do not support CPU-only training."
            )
        self.losses = torch.empty(world_size + 1)
        mp.spawn(
            self.run_rank, args=(world_size,), nprocs=world_size, join=True
        )
        with open(self.pickle_save, "rb") as f:
            self.print(f"Loading pickle from {self.pickle_save}...", end="")
            self = pickle.load(f)
            self.print(
                f"SUCCESS...deleting self.pickle_save...={self.pickle_save}",
                end="",
            )
            os.remove(self.pickle_save)
            self.print("SUCCESS")
        return self


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


def define_names(*tensor_names):
    def decorator(cls):
        def new_init(self, *, data_save, fig_save, verbose=1, **kw):
            super(cls, self).__init__(
                data_save=data_save,
                fig_save=fig_save,
                verbose=verbose,
                tensor_names=list(tensor_names),
                **kw,
            )

        cls.__init__ = new_init
        return cls

    return decorator
