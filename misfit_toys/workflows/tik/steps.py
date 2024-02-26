from misfit_toys.utils import taper


def taper_only(*, length=None, num_batches=None, scale=1.0):
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
