import torch
from deepwave import scalar


class SeismicProp(torch.nn.Module):
    def __init__(
        self,
        *,
        model,
        dx,
        dt,
        freq,
        src_amp_y,
        src_loc_y,
        rec_loc_y,
        src_amp_x=None,
        src_loc_x=None,
        rec_loc_x=None
    ):
        super().__init__()
        self.model = model
        self.dx = dx
        self.dt = dt
        self.freq = freq
        self.src_amp_y = src_amp_y
        self.src_loc_y = src_loc_y
        self.rec_loc_y = rec_loc_y
        self.src_amp_x = src_amp_x
        self.src_loc_x = src_loc_x
        self.rec_loc_x = rec_loc_x

    def forward(self, dummy):
        v = self.model()
        return scalar(
            v,
            self.dx,
            self.dt,
            source_amplitudes=self.src_amp_y,
            source_locations=self.src_loc_y,
            receiver_locations=self.rec_loc_y,
            max_vel=2500,
            pml_freq=self.freq,
            time_pad_frac=0.2,
        )
