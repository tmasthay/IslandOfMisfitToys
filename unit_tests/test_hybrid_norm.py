import sys
import pytest
import os
import numpy as np

sys.path.append('/'.join(os.getcwd().split('/')[:-1]))
from misfits.l1_l2_hybrid import hybrid_norm
# from misfits import hybrid_norm

def __get_test(f,g,delta,tau=1e-8):
    xprmnt = hybrid_norm(f,g,delta)
    def helper(gtruth):
        return np.abs(xprmnt - gtruth) <= tau
    return helper

def __perform_test(f,g,delta,gtruth,tau=1e-8):
    curr = __get_test(f,g,delta,tau)
    assert curr(gtruth(f,g,delta))

def test_hybrid_norm_zero():
    __perform_test(
        f=np.zeros(10),
        g=np.zeros(10),
        delta=1,
        gtruth=lambda a,b,d : 0.0,
        tau=1e-8
    )

def test_hybrid_norm_large_difference():
    __perform_test(
        f=np.array([1,2,3]),
        g=np.array([1000,2000,3000]),
        delta=2,
        gtruth=lambda a,b,d : np.sum(np.abs(a-b)/d),
        tau=1e+1
    )

def test_hybrid_norm_small_difference():
    __perform_test(
        f=np.array([1,2,3]),
        g=np.array([1.1,2.2,3.3]),
        delta=2,
        gtruth=lambda a,b,d :np.sum( 0.5 * ((a-b)/d)**2),
        tau=1e-4
    )
