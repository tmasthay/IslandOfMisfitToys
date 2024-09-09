"""
TODO: Deprecate this module after termining that it is not used.
"""

import torch


def delay_opt(opt_type, **kw):
    """
    TODO: Deprecate if not used.
    """

    def helper(params):
        return opt_type(params, **kw)

    return helper
