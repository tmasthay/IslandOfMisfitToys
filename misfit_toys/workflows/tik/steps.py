import sys
from time import time

import torch

from misfit_toys.utils import taper


def taper_only(*, length=None, num_batches=None, scale=1.0):
    """
    Applies tapering to the output of a neural network model.

    Args:
        length (int, optional): The length of the taper. If None, no tapering is applied.
            If a float between 0 and 1, it is interpreted as a fraction of the output length.
            If an integer greater than 1, it is interpreted as an absolute length.
            Defaults to None.
        num_batches (int, optional): The number of batches. Not yet implemented.
            Defaults to None.
        scale (float, optional): The scaling factor for the loss. Defaults to 1.0.

    Returns:
        helper (function): A helper function that applies tapering to the output of a neural network model.
    """
    if num_batches is not None:
        raise NotImplementedError("num_batches > 1 not yet implemented")

    def helper(self):
        nonlocal length, scale, num_batches
        self.out = self.prop(None)[-1]

        if length is not None:
            if length <= 0:
                raise ValueError("taper length must be positive")
            elif length < 1.0:
                length = int(length * self.out.shape[-1])
            self.out = taper(self.out, length=length)
            obs_data_filt = taper(self.obs_data, length=length)
        else:
            obs_data_filt = self.obs_data

        self.loss = scale * self.loss_fn(self.out, obs_data_filt)
        self.loss.backward()

    return helper


def taper_batch(*, length=None, batch_size=1, scale=1.0, verbose=True):
    """
    Applies tapering to the output of a neural network model.

    Args:
        length (int, optional): The length of the taper. If None, no tapering is applied.
            If a float between 0 and 1, it is interpreted as a fraction of the output length.
            If an integer greater than 1, it is interpreted as an absolute length.
            Defaults to None.
        num_batches (int, optional): The number of batches. Not yet implemented.
            Defaults to None.
        scale (float, optional): The scaling factor for the loss. Defaults to 1.0.

    Returns:
        helper (function): A helper function that applies tapering to the output of a neural network model.
    """
    num_calls = 0

    def helper(self):
        nonlocal length, scale, batch_size, num_calls
        start_time = time()
        if verbose:
            num_calls += 1
            print(f"Call {num_calls}...", flush=True, end='')
        num_shots = self.obs_data.shape[0]
        num_batches = -(-num_shots // batch_size)
        slices = [
            slice(i * batch_size, min((i + 1) * batch_size, num_shots))
            for i in range(num_batches)
        ]

        self.loss = 0.0
        # self.out = torch.zeros_like(self.obs_data)
        epoch_loss = 0.0
        for _, s in enumerate(slices):
            self.optimizer.zero_grad()
            # if verbose:
            #     print(f"Batch {s.start // batch_size + 1}/{num_batches}", flush=True, end='\r')
            # self.out[s] = self.prop(s)[-1]
            self.out = self.prop(s)[-1]

            if length is not None:
                if length <= 0:
                    raise ValueError("taper length must be positive")
                elif length < 1.0:
                    length = int(length * self.out.shape[-1])
                # self.out[s] = taper(self.out[s], length=length)
                self.out = taper(self.out, length=length)
                obs_data_filt = taper(self.obs_data[s], length=length)
            else:
                obs_data_filt = self.obs_data[s]

            lcl_loss = scale * self.loss_fn(self.out, obs_data_filt)
            epoch_loss += lcl_loss.item()
            # epoch_loss += self.loss.item()
            # if verbose:
            #     print(f"Batch {s.start // batch_size + 1}/{num_batches}: Loss={self.loss}", flush=True)
            lcl_loss.backward()
            self.optimizer.step()
        # self.loss = epoch_loss
        self.loss = epoch_loss

        if verbose:
            total_time = time() - start_time
            print(f"took {total_time:.2f} seconds", flush=True)

    return helper
