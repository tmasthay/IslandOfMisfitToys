import numpy as np
from misfits.w2 import w2
from helpers.typlotlib import *
import copy
from scipy.integrate import cumulative_trapezoid

def test_w2_debug():
    N = 100
    i1 = 50
    i2 = 75
    x = np.linspace(0,1,N)
    dx = x[1] - x[0]
    f = np.zeros(N)
    g = np.zeros(N)
    f[i1] = 1.0 / N
    g[i2] = 1.0 / N
    f /= np.trapz(f, dx=dx)
    g /= np.trapz(g, dx=dx)
    F = cumulative_trapezoid(f,dx=dx,initial=0.0).astype(np.float32)
    val, Q, G, U, integrand = w2(f,g,dx,debug=True)

    setup_gg_plot('black', 'black')
    
    plt.plot(x,f, label=r'$f(x)$', linestyle=':', color='red')
    plt.plot(x,g, label=r'$g(x)$', linestyle='-', color='blue')
    set_color_plot(use_legend=True)
    plt.savefig('debug_plots/w2/densities.pdf')
    plt.close()

    plt.plot(x, F, label=r'$F(x)$', linestyle=':', color='red')
    plt.plot(x, G, label=r'$G(x)$', linestyle='-', color='blue')
    set_color_plot(use_legend=True)
    plt.savefig('debug_plots/w2/cdf.pdf')
    plt.close()

    p = np.linspace(0,1,1000).astype(np.float32)
    plt.plot(p, Q(p), label=r'$Q(p)$', linestyle=':', color='red')
    set_color_plot(use_legend=True)
    plt.savefig('debug_plots/w2/compose.pdf')
    plt.close()
    assert 0 == 0

def test_w2_same_distribution():
    N = 100
    dt = 1.0 / N
    tol = 0.0
    f = np.random.random(100) + 0.5
    f /= np.trapz(f, dx=dt)
    g = copy.copy(f)
    assert w2(f, g, dt, tol) == 0.0

def test_w2_dirac():
    N = 100
    i1 = 50
    i2 = 75
    x = np.linspace(0,1,N)
    dx = x[1] - x[0]
    f = np.zeros(N)
    g = np.zeros(N)
    f[i1] = 1.0 / N
    g[i2] = 1.0 / N
    f /= np.trapz(f, dx=dx)
    g /= np.trapz(g, dx=dx)
    val = w2(f,g,dx)
    truth = (i2-i1)*dx
    tol = 1e-5
    assert np.abs(truth - val) < tol

if( __name__ == "__main__" ):
    test_w2_debug()