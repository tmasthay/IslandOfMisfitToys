import torch
from scipy.signal import butter
from torchaudio.functional import biquad
from ...utils import taper
import numpy as np
import torch.distributed as dist

def summarize_tensor(tensor, *, idt_level=0, idt_str='    ', heading='Tensor'):
    # Compute various statistics
    stats = {
        'shape': tensor.shape,
        'mean': torch.mean(tensor).item(),
        'variance': torch.var(tensor).item(),
        'median': torch.median(tensor).item(),
        'min': torch.min(tensor).item(),
        'max': torch.max(tensor).item(),
        'stddev': torch.std(tensor).item()
    }

    # Prepare the summary string with the desired indentation
    indent = idt_str * idt_level
    summary = [f"{heading}:"]
    for key, value in stats.items():
        summary.append(f"{indent}{idt_str}{key} = {value}")

    return '\n'.join(summary)

def print_tensor(tensor, print_fn=print, print_kwargs={}, **kwargs):
    print_fn(summarize_tensor(tensor, **kwargs), **print_kwargs)

class Training:
    def __init__(self, *, distribution):
        self.distribution = distribution

    def train(self):
        rank = self.distribution.rank

        # Setup optimiser to perform inversion
        loss_fn = torch.nn.MSELoss()

        # Run optimisation/inversion
        n_epochs = 2

        all_freqs = [10, 15, 20, 25, 30]

        for (idx, cutoff_freq) in enumerate(all_freqs):
            sos = butter(
                6, 
                cutoff_freq, 
                fs=1.0/self.distribution.dist_prop.module.dt, 
                output='sos'
            )
            sos = [
                torch.tensor(sosi)
                    .to(self.distribution.dist_prop.module.obs_data.dtype) 
                    .to(rank) 
                for sosi in sos
            ]
            print('\n\n')
            # input(sos)

            def filt(x):
                return biquad(biquad(biquad(x, *sos[0]), *sos[1]), *sos[2])
            
            observed_data_filt = filt(
                self.distribution.dist_prop.module.obs_data
            )

            optimiser = torch.optim.LBFGS(
                self.distribution.dist_prop.module.parameters()
            )

            print_tensor(observed_data_filt, print_fn=input)

            for epoch in range(n_epochs):
                closure_calls = 0
                def closure():
                    nonlocal closure_calls
                    closure_calls += 1
                    optimiser.zero_grad()
                    out = self.distribution.dist_prop.module()
                    out_filt = filt(taper(out, 100))
                    loss = 1e6*loss_fn(out_filt, observed_data_filt)
                    if( closure_calls == 1 ):
                        print(
                            f'Loss={loss.item():.16f}, '
                                f'Freq={cutoff_freq}, '
                                f'Epoch={epoch}, '
                                f'Rank={rank}',
                            flush=True
                        )
                    loss.backward()
                    return loss

                optimiser.step(closure)