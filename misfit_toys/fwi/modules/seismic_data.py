from torchaudio.functional import biquad
from ...utils import *
from typing import Annotated as Ant, Optional as Opt
from scipy.ndimage import gaussian_filter

class SeismicData:
    def __init__(self):
        self.ny, self.nx, self.nt = 600, 250, 300
        self.dy, self.dx, self.dt = 4.0, 4.0, 0.004

        self.n_shots, self.src_per_shot, self.rec_per_shot = 16, 1, 100

        self.freq = 25
        self.peak_time = 1.5 / self.freq

        self.taper_length = 100
        self.filter_freq = 40

        self.get_initials()

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

