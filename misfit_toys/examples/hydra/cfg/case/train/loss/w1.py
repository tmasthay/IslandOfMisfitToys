import torch

from misfit_toys.beta.loss import l1_double, lin_decrease
from misfit_toys.beta.loss import softplus as softplus_loss
from misfit_toys.beta.loss import tik_reg, transform_loss
from misfit_toys.beta.renorm import softplus as softplus_renorm

# dep:
#   mod: ^^misfit_toys.beta.loss

# runtime_func: self.train.loss.dep.mod.tik_reg
# kw:
#   f: self.runtime.data.obs_data
#   base_loss:
#     runtime_func: self.train.loss.dep.mod.transform_loss
#     kw:
#       loss: self.train.loss.dep.mod.l1_double
#       transform:
#         runtime_func: ^^null|misfit_toys.beta.renorm|softplus
#         kw:
#           scale: 10.0
#           t: self.runtime.t
#   model_params: self.runtime.prop.module.vp
#   weights:
#     - 1.0
#     - 0.0
#   # penalty: null
#   reg_sched:
#     runtime_func: self.train.loss.dep.mod.lin_decrease
#     kw:
#       max_calls: self.train.max_iters
#       _min: 0.0


def working_w1(
    obs_data, *, model_params, scale, weights, reg_min, t, max_calls
):
    return tik_reg(
        f=obs_data,
        model_params=model_params,
        base_loss=transform_loss(
            loss=l1_double,
            transform=softplus_loss(scale=scale, t=t),
        ),
        weights=weights,
        reg_sched=lin_decrease(max_calls=max_calls, _min=reg_min),
    )


def riel_transform(*, scale, t, eps):
    def helper(f):
        kernel = torch.flip(t + eps, dims=[-1]) ** (scale - 1)
        fnorm = torch.log(1 + torch.exp(scale * f)) / scale
        u = torch.cumulative_trapezoid(kernel * fnorm, t, dim=-1)
        return u

    return helper


def riel(obs_data, *, model_params, scale, weights, reg_min, t, max_calls, eps):
    return tik_reg(
        f=obs_data,
        model_params=model_params,
        base_loss=transform_loss(
            loss=l1_double,
            transform=riel_transform(eps=eps, scale=scale, t=t),
        ),
        weights=weights,
        reg_sched=lin_decrease(max_calls=max_calls, _min=reg_min),
    )
