# from .seismic_data import SeismicProp
from ...utils import idt_print

import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import pickle


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def place_rank(tensor, rank, world_size):
    if tensor is None:
        return None
    elif not isinstance(tensor, torch.Tensor):
        raise TypeError(
            idt_print(
                'misfit_toys.fwi.modules.distribution.place_rank',
                f'Expected tensor, got {type(tensor).__name__}',
                levels=1,
            )
        )
    return torch.chunk(tensor, world_size)[rank].to(rank)


class Distribution:
    def __init__(self, *, rank, world_size, prop):
        self.rank = rank
        self.world_size = world_size
        self.prop = prop
        self.setup_distribution()

    def setup_distribution(self):
        def pr_tensor(obj):
            return place_rank(obj, self.rank, self.world_size)

        # TODO: This may be a source of the bug.
        #    Make sure pytorch is handling parameters properly.
        def pr_param(obj):
            if obj is None:
                return None

            rg = obj.requires_grad
            new_val = pr_tensor(obj)

            if new_val is None:
                return None
            return torch.nn.Parameter(new_val, requires_grad=rg).to(self.rank)

        self.prop.obs_data = pr_tensor(self.prop.obs_data)
        self.prop.src_loc_y = pr_tensor(self.prop.src_loc_y)
        self.prop.rec_loc_y = pr_tensor(self.prop.rec_loc_y)

        # self.prop.src_amp_y.p = pr_param(self.prop.src_amp_y.p)
        # self.prop.src_amp_x.p = pr_param(self.prop.src_amp_x.p)
        self.prop.src_amp_y = pr_tensor(self.prop.src_amp_y)
        self.prop.src_amp_x = pr_tensor(self.prop.src_amp_x)

        self.prop.vp = self.prop.vp.to(self.rank)
        if self.prop.vs is not None:
            self.prop.vs = self.prop.vs.to(self.rank)
            self.prop.rho = self.prop.rho.to(self.rank)

        # if( self.prop.model == 'acoustic' ):
        #     self.prop.vp.p = pr_param(prop.data.vp.p)
        # else:
        #     self.prop.vs.p = pr_param(prop.data.vs.p)
        #     prop.data.rho.p = pr_param(prop.data.rho.p)
        #     prop.data.src_amp_x.p = pr_param(prop.data.src_amp_x.p)

        # print(list(prop.parameters()))
        self.dist_prop = DDP(self.prop, device_ids=[self.rank])

        # input([(type(e), e.requires_grad) for e in list(self.dist_prop.parameters())])
