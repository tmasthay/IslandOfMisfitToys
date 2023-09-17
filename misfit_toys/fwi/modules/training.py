import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from scipy.signal import butter
from torchaudio.functional import biquad
from ...utils import taper

class Training:
    def __init__(
        self,
        *,
        prop,
        src_amp,
        src_loc,
        rec_loc,
        obs_data,
        dt,
        rank
    ):
        self.train(
            prop=prop,
            src_amp=src_amp,
            src_loc=src_loc,
            rec_loc=rec_loc,
            obs_data=obs_data,
            dt=dt,
            rank=rank
        )

    def train(self, *, prop, src_amp, src_loc, rec_loc, obs_data, dt, rank):
        # Setup optimiser to perform inversion
        loss_fn = torch.nn.MSELoss()

        # Run optimisation/inversion
        n_epochs = 2

        for cutoff_freq in [10, 15, 20, 25, 30]:
            sos = butter(6, cutoff_freq, fs=1/dt, output='sos')
            sos = [torch.tensor(sosi).to(obs_data.dtype).to(rank) for sosi in sos]

            def filt(x):
                return biquad(biquad(biquad(x, *sos[0]), *sos[1]),
                            *sos[2])
            observed_data_filt = filt(obs_data)
            optimiser = torch.optim.LBFGS(prop.parameters())
            for epoch in range(n_epochs):
                def closure():
                    optimiser.zero_grad()
                    out = prop(src_amp, src_loc, rec_loc)
                    out_filt = filt(taper(out[-1], 100))
                    loss = 1e6*loss_fn(out_filt, observed_data_filt)
                    print(
                        f'Rank={rank}, Freq={cutoff_freq}, Epoch={epoch}, ' +
                        f'Loss={loss.item():.4e}',
                        flush=True
                    )
                    loss.backward()
                    return loss

                optimiser.step(closure)