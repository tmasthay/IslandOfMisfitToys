import os
import torch
import torch.distributed as dist
from .models import Prop
from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

class Distribution:
    def __init__(self, *, rank, world_size):
        self.rank = rank
        self.world_size = world_size
        
    def setup_distribution(
        self,
        *,
        obs_data,
        src_amp,
        src_loc,
        rec_loc,
        model,
        dx,
        dt,
        freq
    ):
        #chunk the data according to rank
        obs_data = torch.chunk(
            obs_data, 
            self.world_size
        )[self.rank].to(self.rank)

        src_amp = torch.chunk(
            src_amp, 
            self.world_size
        )[self.rank].to(self.rank)

        src_loc = torch.chunk(
            src_loc, 
            self.world_size
        )[self.rank].to(self.rank)

        rec_loc = torch.chunk(
            rec_loc, 
            self.world_size
        )[self.rank].to(self.rank)

        prop = Prop(model, dx, dt, freq).to(self.rank)
        prop = DDP(prop, device_ids=[self.rank])
        return prop, obs_data, src_amp, src_loc, rec_loc