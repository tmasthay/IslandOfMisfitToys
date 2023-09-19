from ..dataset import DataFactoryMeta, towed_src, fixed_rec
from ...utils import DotDict
import os
import torch
from warnings import warn
import deepwave as dw 

class Factory(DataFactoryMeta):
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