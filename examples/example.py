from misfit_toys.modules.distribution import cleanup, setup

from abc import ABC, abstractmethod
import os
import torch

class Example(ABC):
    def __init__(self, *, data_save, fig_save, tensor_names, **kw):
        self.data_save = data_save
        self.fig_save = fig_save
        self.tensor_names = tensor_names
        self.output_files = [f'{data_save}/{e}.pt' for e in tensor_names]
        if( 'output_files' in kw.keys() ):
            raise ValueError(
                'output_files is generated from tensor_names'
                    'and thus should be treated like a reserved keyword'
            )
        self.__dict__.update(kw)

    @abstractmethod
    def generate_data(self, rank, world_size):
        pass

    @abstractmethod
    def plot_data(self, *, data_save, fig_save, **kw):
        pass

    def run(self, rank, world_size):
        setup(rank, world_size)
        see_files = [os.path.exists(f) for f in self.output_files]
        if( all(see_files) ):
            self.generate_data(rank, world_size)
        else:
            if( rank == 0 ):
                print(
                    'Skipping data generation, delete .pt files in '
                        f'{data_save} and re-run this script to regenerate', 
                    flush=True
                )
        torch.distributed.barrier()
    
    if
        self.generate_data(rank, world_size)
        self.plot_data(data_save=self.data_save, fig_save=self.fig_save)

def MultiscaleExample(Example):
    def generate_data(self, **kw):
        pass

    def plot_data(self, **kw):
        torch.save(vp_record, f'{data_save}/vp_record.pt')
        torch.save(v_init, f'{data_save}/vp_init.pt')
        torch.save(v_true, f'{data_save}/vp_true.pt')
 
        v = vp_record[-1, -1, :, :]
        vmin = v_true.min()
        vmax = v_true.max()
        cmap = 'gray'
        _, ax = plt.subplots(3, figsize=(10.5, 10.5), sharex=True,
                             sharey=True)
        ax[0].imshow(v.T, aspect='auto', cmap=cmap,
                     vmin=vmin, vmax=vmax)
        ax[0].set_title("Initial")
        ax[1].imshow(v.T, aspect='auto', cmap=cmap,
                     vmin=vmin, vmax=vmax)
        ax[1].set_title("Out")
        ax[2].imshow(v_true.T, aspect='auto', cmap=cmap,
                     vmin=vmin, vmax=vmax)
        ax[2].set_title("True")
        plt.tight_layout()
        plt.savefig(f'{fig_save}/deepwave_ddp_final_inversion.jpg')

        for freq_idx,epoch in prod(
            range(vp_record.shape[0]), 
            range(vp_record.shape[1])
        ):
            ax[1].imshow(vp_record[freq_idx,epoch].cpu().T, aspect='auto', cmap=cmap,
                       vmin=vmin, vmax=vmax)
            ax[1].set_title(f"Epoch {epoch}, cutoff freq {freqs[freq_idx]}")
            plt.savefig(f'{fig_save}/deepwave_ddp_{freq_idx}_{epoch}.jpg')
        
        dw_iters = f'{fig_save}/deepwave_ddp_[0-9]*_[0-9]*.jpg'
        cmd = (
            f'convert -delay 100 -loop 0 '
                f'$(ls -tr {dw_iters}) {fig_save}/deepwave_ddp.gif'
        )
        print('GIF creation attempt')
        print(f'    cmd="{cmd}"')
        os.system(cmd)
        os.system(f'rm {dw_iters}')