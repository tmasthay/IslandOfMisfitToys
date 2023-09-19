from ..dataset import *
from ...utils import DotDict
from .metadata import metadata
import os
import torch
from warnings import warn
import deepwave as dw 

class Factory(DataFactory):
    def __init__(self, *, path, device=None):
        super().__init__(path=path)
        if( device is None ):
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        self.metadata = metadata()

    def generate_derived_data(self, *, data):
        warn(
            '\n    This function contains a CUDA memory leak!' 
            '\n    Run this prior to your main script and then use ' 
            '\n    get_data function to pull in obs_data.'
            '\n    Doing so will make this small memory leak only occur once'
            ' and be cleaned up by the garbage collector and thus benign.'
        )
        vp = data['vp'].to(self.device)

        d = DotDict(data)
        d.ny, d.nx = vp.shape

        # n_shots = 115

        # n_sources_per_shot = 1
        # d_source = 20  # 20 * 4m = 80m
        # first_source = 10  # 10 * 4m = 40m
        # source_depth = 2  # 2 * 4m = 8m

        # n_receivers_per_shot = 384
        # d_receiver = 6  # 6 * 4m = 24m
        # first_receiver = 0  # 0 * 4m = 0m
        # receiver_depth = 2  # 2 * 4m = 8m

        # freq = 25
        # nt = 750
        # dt = 0.004
        # peak_time = 1.5 / freq

        # src_loc = torch.zeros(
        #     d.n_shots, 
        #     d.n_sources_per_shot, 
        #     2,
        #     dtype=torch.long
        # ).to(self.device)
        # src_loc[..., 1] = d.src_depth
        # src_loc[:, 0, 0] = torch.arange(d.n_shots) * d.d_src + d.fst_src

        # # receiver_locations
        # rec_loc = torch.zeros(d.n_shots, d.rec_per_shot, 2, dtype=torch.long) \
        #     .to(self.device)
        # rec_loc[..., 1] = d.rec_depth
        # rec_loc[:, :, 0] = (
        #     (torch.arange(n_receivers_per_shot) * d_receiver +
        #     first_receiver)
        #     .repeat(n_shots, 1)
        # )
        src_loc = towed_src(
            n_shots=d.n_shots,
            src_per_shot=d.src_per_shot,
            d_src=d.d_src,
            fst_src=d.fst_src,
            src_depth=d.src_depth,
            d_intra_shot=d.d_intra_shot
        ).to(self.device)

        rec_loc = fixed_rec(
            n_shots=d.n_shots,
            rec_per_shot=d.rec_per_shot,
            d_rec=d.d_rec,
            fst_rec=d.fst_rec,
            rec_depth=d.rec_depth
        ).to(self.device)

        # source_amplitudes
        src_amp = \
            dw.wavelets.ricker(d.freq, d.nt, d.dt, d.peak_time) \
            .repeat(d.n_shots, d.src_per_shot, 1) \
            .to(self.device)
        
        print('Building obs_data')
        out = dw.scalar(
            vp,
            d.dx,
            d.dt,
            source_amplitudes=src_amp,
            source_locations=src_loc,
            receiver_locations=rec_loc,
            pml_freq=d.freq,
            accuracy=d.accuracy
        )[-1]
        out_cpu = out.to('cpu')

        outputs = {
            'obs_data': out.to('cpu'),
            'src_amp': src_amp.to('cpu'),
            'src_loc': src_loc.to('cpu'),
            'rec_loc': rec_loc.to('cpu')
        }
        for k,v in outputs.items():
            print(f'Saving {k}...', end='')
            torch.save(v, os.path.join(self.path, f'{k}.pt'))
            print('SUCCESS')

        del src_amp, src_loc, rec_loc, vp
        del out, out_cpu
        torch.cuda.empty_cache()

    def manufacture_data(self):
        self._manufacture_data(metadata=self.metadata)