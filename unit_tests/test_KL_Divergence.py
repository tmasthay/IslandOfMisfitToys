import sys
import pytest
import os
import numpy as np

sys.path.append('/'.join(os.getcwd().split('/')[:-1]))
from misfits.KL_Divergence import KL_Divergence
# from misfits import hybrid_norm

def __get_test(f,g,delta,tau=1e-8):
    xprmnt = KL_Divergence(f,g)
    def helper(gtruth):
        return np.abs(xprmnt - gtruth) <= tau
    return helper

def __perform_test(f,g,delta,gtruth,tau=1e-8):
    curr = __get_test(f,g,delta,tau)
    assert curr(gtruth(f,g,delta))

def test_kl_divergence_zero():
    __perform_test(
        f=np.array([0.3, 0.4, 0.3]),
        g=np.array([0.3, 0.4, 0.3]),
        gtruth=lambda a,b,d : 0.0,
        tau=1e-8
    )

