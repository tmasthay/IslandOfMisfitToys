import torch


def delay_opt(opt_type, **kw):
    def helper(params):
        return opt_type(params, **kw)

    return helper
