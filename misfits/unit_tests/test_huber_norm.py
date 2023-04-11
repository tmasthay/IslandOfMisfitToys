import sys
import pytest
import os
import numpy as np

sys.path.append(os.path.abspath(".."))
from huber import huber_norm

def __get_test(f,g,delta,tau=1e-8):
    xprmnt = huber_norm(f,g,delta)
    def helper(gtruth):
        return np.abs(xprmnt - gtruth) <= tau
    return helper

def __perform_test(f,g,delta,gtruth,tau=1e-8):
    curr = __get_test(f,g,delta,tau)
    assert curr(gtruth(f,g,delta))

def test_huber_norm_basic():
    __perform_test(
        f=np.array([1,2,3]),
        g=np.array([2,3,4]),
        delta=0.5,
        gtruth=lambda a,b,d : d * (np.mean(np.abs(a-b)) - 0.5 * d),
        tau=1e-8
    )

def test_huber_norm_large_delta():
    __perform_test(
        f=np.array([1,2,3]),
        g=np.array([2,3,4]),
        delta=100.0,
        gtruth=lambda a,b,d : 0.5 * np.mean((a-b)**2),
        tau=1e-8
    )

def test_huber_norm_same_values():
    __perform_test(
        f=np.array([1,2,3]),
        g=np.array([1,2,3]),
        delta=0.5,
        gtruth=lambda a,b,d : 0.0,
        tau=1e-8
    )

def test_huber_norm_large_difference():
    __perform_test(
        f=np.array([1,2,3]),
        g=np.array([100,200,300]),
        delta=0.5,
        gtruth=lambda a,b,d : d * (np.mean(np.abs(a-b)) - 0.5 * d),
        tau=1e-8
    )

