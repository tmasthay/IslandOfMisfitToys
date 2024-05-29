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
        raise NotImplementedError("nonzero num_batches not yet implemented")

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
