from time import time

from misfit_toys.utils import taper


def taper_only(*, length=None, num_batches=None, scale=1.0, src_amp_y, **kw):
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
        self.out = self.prop(None, src_amp_y(), **kw)[-1]

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


def taper_batch(
    *, length=None, batch_size=1, scale=1.0, verbose=False, src_amp_y, **kw
):
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
        num_calls += 1
        start_time = time()
        if verbose:
            print(f"    taper_batch: call == {num_calls}", flush=True, end="")
        num_shots = self.obs_data.shape[0]
        num_batches = -(-num_shots // batch_size)
        slices = [
            slice(i * batch_size, min((i + 1) * batch_size, num_shots))
            for i in range(num_batches)
        ]

        self.loss = 0.0
        epoch_loss = 0.0
        amps = src_amp_y()
        for _, s in enumerate(slices):
            self.out = self.prop(s, amps, **kw)[-1]

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

            self.loss = scale * self.loss_fn(self.out, obs_data_filt)
            epoch_loss += self.loss
            self.loss.backward()
        self.loss = epoch_loss / num_batches
        if verbose:
            print(f"...took {time() - start_time}s", flush=True)

    return helper
