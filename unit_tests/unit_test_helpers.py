import torch

def check(res, expct, tau=1e-5, dst=lambda u,v: torch.abs(u-v)):
    u = dst(res, expct)
    pass_test = u <= tau
    return pass_test, "(Real, Desired) = (%.4e, %.4e)"%(u, tau)
