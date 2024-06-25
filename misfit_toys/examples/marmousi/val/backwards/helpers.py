import torch
from deepwave import scalar

# Generate a velocity model constrained to be within a desired range
class Model(torch.nn.Module):
    def __init__(self, initial, min_vel, max_vel):
        super().__init__()
        self.min_vel = min_vel
        self.max_vel = max_vel
        self.model = torch.nn.Parameter(
            torch.logit((initial - min_vel) / (max_vel - min_vel))
        )

    def forward(self):
        return (
            torch.sigmoid(self.model) * (self.max_vel - self.min_vel)
            + self.min_vel
        )
    
class Prop(torch.nn.Module):
    def __init__(self, *, model, dx, dt, freq, source_amplitudes, source_locations, receiver_locations):
        super().__init__()
        self.model = model
        self.dx = dx
        self.dt = dt
        self.freq = freq
        self.source_amplitudes = source_amplitudes
        self.source_locations = source_locations
        self.receiver_locations = receiver_locations

    def forward(self, slicer):
        v = self.model()
        if slicer is None:
            slicer = slice(None)
        return scalar(
            v,
            self.dx,
            self.dt,
            source_amplitudes=self.source_amplitudes[slicer],
            source_locations=self.source_locations[slicer],
            receiver_locations=self.receiver_locations[slicer],
            max_vel=2500,
            pml_freq=self.freq,
            time_pad_frac=0.2,
        )