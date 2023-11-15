import torch
from deepwave import scalar, elastic
from deepwave.common import vpvsrho_to_lambmubuoyancy as get_lame
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
        vp,
        vs=None,
        rho=None,
        model='acoustic',
        dx,
        dt,
        src_amp_y=None,
        src_loc_y=None,
        rec_loc_y=None,
        src_amp_x=None,
        src_loc_x=None,
        rec_loc_x=None,
        **kw,
    ):
        super().__init__()
        self.vp = vp
        self.vs = vs
        self.rho = rho
        self.model = model.lower()
        self.dx = dx
        self.dt = dt
        self.src_amp_y = src_amp_y
        self.src_loc_y = src_loc_y
        self.rec_loc_y = rec_loc_y
        self.src_amp_x = src_amp_x
        self.src_loc_x = src_loc_x
        self.rec_loc_x = rec_loc_x
        self.extra_forward = kw
        self.__validate_init__()

    def __validate_init__(self):
        if self.model not in ['acoustic', 'elastic']:
            raise ValueError(
                f"model must be 'acoustic' or 'elastic', not {self.model}"
            )
        y_set = not any(
            [
                e is None
                for e in [self.src_amp_y, self.rec_loc_y, self.src_loc_y]
            ]
        )
        x_set = not any(
            [
                e is None
                for e in [self.src_amp_x, self.rec_loc_x, self.src_loc_x]
            ]
        )
        if not (y_set or x_set) or ((not y_set) and self.model == 'acoustic'):
            raise ValueError(
                'acoustic model requires y set, elastic y or x set'
            )

    def __get_optional_param__(self, name):
        if getattr(self, name) is None:
            return None
        else:
            return getattr(self, name)()

    def forward(self, dummy):
        if self.model.lower() == 'acoustic':
            return scalar(
                self.vp(),
                self.dx,
                self.dt,
                source_amplitudes=self.src_amp_y(),
                source_locations=self.src_loc_y,
                receiver_locations=self.rec_loc_y,
                **self.extra_forward,
            )
        elif self.model.lower() == 'elastic':
            lame_params = get_lame(self.vp(), self.vs(), self.rho())
            return elastic(
                *lame_params,
                self.dx,
                self.dt,
                source_amplitudes_y=self.__get_optional_param__('src_amp_y'),
                source_locations_y=self.__get_optional_param__('src_loc_y'),
                receiver_locations_y=self.__get_optional_param__('rec_loc_y'),
                source_amplitudes_x=self.__get_optional_param__('src_amp_x'),
                source_locations_x=self.__get_optional_param__('src_loc_x'),
                receiver_locations_x=self.__get_optional_param__('rec_loc_x'),
                **self.extra_forward,
            )
