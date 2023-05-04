import numpy as np
import os,sys

sys.path.append('/'.join(os.getcwd().split('/')[:-1]))
from misfits.w2 import w2

def test_w2_same_distribution():
    f = np.array([0.1, 0.3, 0.6, 0.0], dtype=np.float32)
    g = np.array([0.1, 0.3, 0.6, 0.0], dtype=np.float32)
    dt = 0.1
    tol = 0.0
    assert w2(f, g, dt, tol) == 0.0

def test_w2_different_distributions():
    f = np.array([0.1, 0.3, 0.6, 0.0], dtype=np.float32)
    g = np.array([0.0, 0.4, 0.4, 0.2], dtype=np.float32)
    dt = 0.1
    tol = 0.0
    assert w2(f, g, dt, tol) == 0.016489159

def test_w2_different_distributions_with_tolerance():
    f = np.array([0.1, 0.3, 0.6, 0.0], dtype=np.float32)
    g = np.array([0.0, 0.4, 0.4, 0.2], dtype=np.float32)
    dt = 0.1
    tol = 1e-6
    assert w2(f, g, dt, tol) == 0.016489159

def test_w2_large_distributions():
    f = np.random.rand(10000).astype(np.float32)
    f /= np.sum(f)
    g = np.random.rand(10000).astype(np.float32)
    g /= np.sum(g)
    dt = 0.01
    tol = 0.0
    assert w2(f, g, dt, tol) > 0.0

