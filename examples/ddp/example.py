from misfit_toys.fwi.modules.distribution import cleanup, setup

from abc import ABC, abstractmethod
import os
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import matplotlib.pyplot as plt
from warnings import warn

# config -- ignore


class Example(ABC):
    def __init__(self, *, data_save, fig_save, tensor_names, verbose=1, **kw):
        self.data_save = data_save
        self.fig_save = fig_save
        self.tensor_names = tensor_names
        self.output_files = {
            e: os.path.join(data_save, f"{e}.pt") for e in self.tensor_names
        }
        self.verbose = verbose
        self.tensors = {}
        if "output_files" in kw.keys():
            raise ValueError(
                "output_files is generated from tensor_names"
                "and thus should be treated like a reserved keyword"
            )
        self.__dict__.update(kw)

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

        _, ax = plt.subplots(3, **subplot_args)
        Example.print_static(
            f"Plotting final results {name} at {fig_save}/final_{name}.jpg...",
            end="",
            verbose=verbose,
        )
        ax[0].imshow(init, **plot_args)
        ax[0].set_title(f"{name} Initial")
        ax[1].imshow(reshaped_record[-1], **plot_args)
        ax[1].set_title(f"{name} Final")
        ax[2].imshow(true, **plot_args)
        ax[2].set_title(f"{name} Ground Truth")
        ax[1].cla()
        Example.print_static("SUCCESS", verbose=verbose)
        for i, curr in enumerate(reshaped_record):
            save_file_name = f"{fig_save}/{name}_{i}.jpg"
            Example.print_static(
                f"Plotting {save_file_name} with params={subtitle(i)}...",
                end="",
            )
            ax[1].imshow(curr, **plot_args)
            ax[1].set_title(f"{name} {subtitle(i)}")
            plt.tight_layout()
            plt.savefig(save_file_name)
            ax[1].cla()
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
        self, *, name, labels, subplot_args=None, plot_args=None
    ):
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
        if rank == 0:
            self.save_all_tensors()
        torch.distributed.barrier()

    def plot_field(self, *, field, **kw):
        raise NotImplementedError(
            "plot_field not implemented, please override in subclass if you"
            " intend to use it"
        )

    def print(self, *args, level=1, **kw):
        Example.print_static(*args, level=level, verbose=self.verbose, **kw)

    def save_tensor(self, name):
        torch.save(self.tensors[name], self.output_files[name])

    def save_all_tensors(self):
        if set(self.tensors.keys()) != set(self.tensor_names):
            raise ValueError(
                "FATAL: tensor_names and self.tensors.keys() do not match.\n"
                f"    tensor_names: {self.tensor_names}\n"
                f"    self.tensors.keys(): {self.tensors.keys()}"
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
        full_loss = self.tensors["loss"].sum(dim=0)
        for r in range(world_size):
            label = f"rank {r}"
            plt.plot(
                self.tensors["loss"][r],
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

    def plot_fields(self, **kw):
        for fld, tnsr in self.tensors.items():
            self.plot_field(field=fld, tensor=tnsr, **kw)
        self.plot_loss(**kw)

    def run_rank(self, rank, world_size):
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
