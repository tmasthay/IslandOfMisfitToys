import torch


def build_simple(*, type, **kw):
    def helper(params):
        return type(params, **kw)

    return helper
