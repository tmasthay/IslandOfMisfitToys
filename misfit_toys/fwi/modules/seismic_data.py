from torchaudio.functional import biquad
from ...utils import *
from typing import Annotated as Ant, Optional as Opt
from scipy.ndimage import gaussian_filter

class SeismicData:
    def __init__(self):
        self.v_true, self.v_init, self.obs_data = self.get_initials(
            ny=600,
            nx=250,
            filter_freq=40.0,
            n_shots=16,
            rec_per_shot=100,
            nt=300
        )

    def get_initials(self, *, ny, nx, filter_freq, n_shots, rec_per_shot, nt):
        #grab marmousi data
        v_true = get_data(field='vp', folder='marmousi', path='conda')
        obs_data = get_data(field='obs_data', folder='marmousi', path='conda')
        obs_data = taper(obs_data[:n_shots, :rec_per_shot, :nt], 100)

        # Select portion of model for inversion
        ny = 600
        nx = 250
        v_true = v_true[:ny, :nx]

        # Smooth to use as starting model
        v_init = torch.tensor(1/gaussian_filter(1/v_true.numpy(), filter_freq))

        return v_true, v_init, obs_data
