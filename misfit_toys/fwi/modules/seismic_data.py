from ...utils import *
from torchaudio.functional import biquad
from typing import Annotated as Ant, Optional as Opt
from scipy.ndimage import gaussian_filter

class SeismicData:
    def __init__(
        self,
        *,
        ny,
        nx,
        nt,
        dy,
        dx,
        dt,
        n_shots,
        src_per_shot,
        rec_per_shot,
        d_src,
        fst_src,
        src_depth,
        d_rec,
        fst_rec,
        rec_depth,
        d_intra_shot,
        freq,
        peak_time,
        taper_length,
        filter_freq
    ):
        self.ny, self.nx, self.nt = ny, nx, nt
        self.dy, self.dx, self.dt = dy, dx, dt

        self.n_shots, self.src_per_shot, self.rec_per_shot = \
            n_shots, src_per_shot, rec_per_shot
        self.d_src, self.fst_src, self.src_depth = d_src, fst_src, src_depth
        self.d_rec, self.fst_rec, self.rec_depth = d_rec, fst_rec, rec_depth
        self.d_intra_shot = d_intra_shot

        self.freq = freq
        self.peak_time = peak_time

        self.taper_length = taper_length
        self.filter_freq = filter_freq
        
        self.get_initials()
        self.build_survey()

    def get_initials(self):
        #grab marmousi data
        self.v_true = get_data(field='vp', folder='marmousi', path='conda')
        self.obs_data = get_data(
            field='obs_data', 
            folder='marmousi', 
            path='conda'
        )
        self.obs_data = taper(
            self.obs_data[:self.n_shots, :self.rec_per_shot, :self.nt], 
            self.taper_length
        )

        self.v_true = self.v_true[:self.ny, :self.nx]

        # Smooth to use as starting model
        self.v_init = torch.tensor(
            1/gaussian_filter(1/self.v_true.numpy(), self.filter_freq)
        )

    def build_survey(self):
        self.src_loc = towed_src(
            n_shots=self.n_shots,
            src_per_shot=self.src_per_shot,
            d_src=self.d_src,
            fst_src=self.fst_src,
            src_depth=self.src_depth,
            d_intra_shot=self.d_intra_shot
        )

        #receiver locations
        self.rec_loc = fixed_rec(
            n_shots=self.n_shots,
            rec_per_shot=self.rec_per_shot,
            d_rec=self.d_rec,
            rec_depth=self.rec_depth,
            fst_rec=self.fst_rec
        )

        # source amplitudes
        self.src_amp_y = \
            dw.wavelets.ricker(self.freq, self.nt, self.dt, self.peak_time) \
            .repeat(self.n_shots, self.src_per_shot, 1)
    # )

