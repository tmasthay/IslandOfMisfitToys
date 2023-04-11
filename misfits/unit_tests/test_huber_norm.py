import sys
import pytest
import os
import numpy as np

sys.path.append(os.path.abspath(".."))
from huber import huber_norm

tau = 1e-8  # Tolerance value

def test_huber_norm_basic():
    f = np.array([1, 2, 3])
    g = np.array([2, 3, 4])
    delta = 0.5
    assert np.abs(huber_norm(f, g, delta) - 0.5 * np.mean((f - g)**2)) < tau

def test_huber_norm_large_delta():
    f = np.array([1, 2, 3])
    g = np.array([2, 3, 4])
    delta = 100
    assert np.abs(huber_norm(f, g, delta) - 0.5 * np.mean((f - g)**2)) < tau

def test_huber_norm_same_values():
    f = np.array([1, 2, 3])
    g = np.array([1, 2, 3])
    delta = 0.5
    assert np.abs(huber_norm(f, g, delta) - 0) < tau

def test_huber_norm_large_difference():
    f = np.array([1, 2, 3])
    g = np.array([100, 200, 300])
    delta = 0.5
    expected_loss = delta * (np.abs(f - g) - 0.5 * delta)
    assert np.abs(huber_norm(f, g, delta) - np.mean(expected_loss)) < tau

