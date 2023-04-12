import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pytest
from misfits.sobolev import sobolev_norm as sobolev
import os
from typlotlib import *

def test_sobolev_order_0():
    # Test the sobolev seminorm of order 0 for a simple functionç
    # f = lambda x: 1.0 if x >= -1.0 and x <= 1.0 else 0.0
    alpha = 2.0
    f = lambda x : np.real(np.exp(1j * 2 * np.pi * alpha * x))
    T = 10.0 * 2 * np.pi
    ot = -T
    nt = 10000
    t = np.linspace(ot, T, nt)
    dt = t[1] - t[0]
    F = np.array(list(map(f, t)))
    s = 0
    result, xi, G = sobolev(F, s, ot=ot, dt=dt, nt=nt)
    
    c1 = rand_color()
    c2 = rand_color()
    setup_gg_plot(c1, c2)

    plt.suptitle('%s : %s'%(c1, c2))
    
    plt.subplot(2,1,1)
    plt.plot(
        xi, 
        G, 
        color='blue', 
        label=r'$Re\left[\hat{f}(\xi)\right]^{2}$'
    )
    set_color_plot(xlabel=r'$\xi$', ylabel=r'$\hat{f}(\xi)$', use_legend=True)
    plt.legend()

    plt.subplot(2,1,2)
    plt.plot(t, np.abs(F)**2, color='blue')
    set_color_plot(xlabel=r'$t$', ylabel=r'$f(t)^{2}$')

    os.system('mkdir -p sobolev_plots')
    plt.savefig('sobolev_plots/integrand.pdf')
    expected = np.trapz(F**2, dx=t[1]-t[0])
    tau = 1e-1
    assert np.abs(expected - result) <= tau, '%.4e, %.4e'%(result, expected)
