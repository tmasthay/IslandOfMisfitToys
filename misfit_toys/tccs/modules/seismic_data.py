import torch
from deepwave import scalar
from masthay_helpers.global_helpers import DotDict


class Param(torch.nn.Module):
    def __init__(self, *, p, requires_grad=False, **kw):
        super().__init__()
        self.p = torch.nn.Parameter(p, requires_grad=requires_grad)
        self.custom = DotDict(kw)

    def forward(self):
        return self.p

    @classmethod
    def delay_init(cls, **kw):
        return lambda p: cls(p=p, **kw)


class ParamConstrained(Param):
    def __init__(self, *, p, minv, maxv, requires_grad=False):
        super().__init__(
            p=torch.logit((p - minv) / (maxv - minv)),
            requires_grad=requires_grad,
            minv=minv,
            maxv=maxv,
        )

    def forward(self):
        minv = self.custom.minv
        maxv = self.custom.maxv
        return torch.sigmoid(self.p) * (maxv - minv) + minv


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
        rec_loc_x=None,
        extra_forward=None,
        **kw
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
        self.extra_forward = extra_forward or {}
        self.__dict__.update(kw)

    def forward(self, dummy):
        v = self.model()
        return scalar(
            v,
            self.dx,
            self.dt,
            source_amplitudes=self.src_amp_y,
            source_locations=self.src_loc_y,
            receiver_locations=self.rec_loc_y,
            **self.extra_forward,
        )
