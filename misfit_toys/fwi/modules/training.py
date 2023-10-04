import torch
from scipy.signal import butter
from torchaudio.functional import biquad
from ...utils import taper
import numpy as np
import torch.distributed as dist

from .distribution import cleanup

import os


def summarize_tensor(tensor, *, idt_level=0, idt_str='    ', heading='Tensor'):
    # Compute various statistics
    stats = {
        'shape': tensor.shape,
        'mean': torch.mean(tensor).item(),
        'variance': torch.var(tensor).item(),
        'median': torch.median(tensor).item(),
        'min': torch.min(tensor).item(),
        'max': torch.max(tensor).item(),
        'stddev': torch.std(tensor).item(),
    }

    # Prepare the summary string with the desired indentation
    indent = idt_str * idt_level
    summary = [f"{heading}:"]
    for key, value in stats.items():
        summary.append(f"{indent}{idt_str}{key} = {value}")

    return '\n'.join(summary)


def print_tensor(tensor, print_fn=print, print_kwargs=None, **kwargs):
    if print_kwargs is None:
        print_kwargs = {'flush': True}
    print_fn(summarize_tensor(tensor, **kwargs), **print_kwargs)


class Training:
    def __init__(self, *, dist_prop, rank, world_size):
        self.dist_prop = dist_prop
        self.rank = rank
        self.world_size = world_size

    def train(self, *, path, **kw):
        # Setup optimiser to perform inversion
        loss_fn = torch.nn.MSELoss()

        # Run optimisation/inversion
        n_epochs = 2

        all_freqs = torch.Tensor([10, 15, 20, 25, 30])
        n_freqs = all_freqs.shape[0]
        freqs = all_freqs
        loss_local = torch.zeros(freqs.shape[0], n_epochs).to(self.rank)
        vp_record = torch.Tensor(
            n_freqs, n_epochs, *self.dist_prop.module.vp.p.shape
        )

        print(
            f'enumerate(all_freq)={[e for e in enumerate(all_freqs)]}',
            flush=True,
        )
        for idx, cutoff_freq in enumerate(list(all_freqs)):
            sos = butter(
                6,
                cutoff_freq,
                fs=1.0 / self.dist_prop.module.dt,
                output='sos',
            )
            sos = [
                torch.tensor(sosi)
                .to(self.dist_prop.module.obs_data.dtype)
                .to(self.rank)
                for sosi in sos
            ]
            # input(sos)

            def filt(x):
                return biquad(biquad(biquad(x, *sos[0]), *sos[1]), *sos[2])

            observed_data_filt = filt(self.dist_prop.module.obs_data)

            optimiser = torch.optim.LBFGS(self.dist_prop.module.parameters())

            # print_tensor(observed_data_filt, print_fn=input)

            for epoch in range(n_epochs):
                closure_calls = 0

                def closure():
                    nonlocal closure_calls, loss_local
                    closure_calls += 1
                    optimiser.zero_grad()
                    # out = self.distribution.dist_prop.module(**kw)
                    out = self.dist_prop(1, **kw)
                    out_filt = filt(taper(out[-1], 100))
                    loss = 1e6 * loss_fn(out_filt, observed_data_filt)
                    if closure_calls == 1:
                        print(
                            (
                                f'Loss={loss.item():.16f}, '
                                f'Freq={cutoff_freq}, '
                                f'Epoch={epoch}, '
                                f'Rank={self.rank}'
                            ),
                            flush=True,
                        )
                        loss_local[idx, epoch] = loss
                    loss.backward()
                    return loss

                optimiser.step(closure)
                vp_record[idx, epoch] = (
                    self.dist_prop.module.vp().detach().cpu()
                )
        os.makedirs(path, exist_ok=True)

        def save(k, v):
            u = v.detach().cpu()
            lcl_path = os.path.join(path, f'{k}_{self.rank}.pt')
            print(f'Saving to {lcl_path}...', flush=True, end='')
            torch.save(u, lcl_path)
            print('SUCCESS', flush=True)

        save('loss', loss_local)
        save('freqs', freqs)
        save('vp_record', vp_record)
