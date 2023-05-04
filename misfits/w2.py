from .helpers.quantile.smart_quantile_cython import smart_quantile_peval as ppf
import numpy as np
from scipy.integrate import cumulative_trapezoid

def w2_peval(f,dt,tol=0.0):
    nt = len(f)
    t = np.array(np.linspace(0.0, (nt-1)*dt, nt), dtype=np.float32)
    Q = ppf(t, f, tol)
    def aux(g):
        G = np.array(
            cumulative_trapezoid(g,dx=dt,initial=0.0), 
            dtype=np.float32
        )
        return np.trapz((Q(G) - t)**2 * g, dx=dt)
    return aux

def w2(f,g,dt,tol=0.0):
    u = w2_peval(f,dt,tol)
    return u(g)


    
