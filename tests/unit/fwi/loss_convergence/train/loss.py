from misfit_toys.fwi.loss import W2Loss
import torch


def w2_generator(*, t, p, renorm):
    def helper(*, obs_data):
        return W2Loss(
            t=t,
            p=p,
            obs_data=obs_data,
            renorm=renorm,
            gen_deriv=(lambda *args, **kwargs: None),
            down=1,
        )

    return helper


class MSEFixed(torch.nn.Module):
    def __init__(self, *, obs_data):
        super().__init__()

        self.obs_data = obs_data

    def forward(self, curr):
        return torch.mean((curr - self.obs_data) ** 2)
