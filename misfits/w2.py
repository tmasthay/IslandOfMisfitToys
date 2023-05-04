from .helpers.quantile.smart_quantile_cython import smart_quantile_peval 
import numpy as np
from scipy.integrate import cumulative_trapezoid

def w2_peval(f,dt,tol=0.0):
    nt = len(f)
    t = np.array(np.linspace(0.0, (nt-1)*dt, nt), dtype=np.float32)
    Q = smart_quantile_peval(f, t, tol)
    def aux(g, debug=False):
        G = np.array(
            cumulative_trapezoid(g,dx=dt,initial=0.0), 
            dtype=np.float32
        )
        integrand = (Q(G) - t)**2 * g
        val = np.trapz(integrand, dx=dt)
        if( debug ):
            return val, Q, G, Q(G), integrand
        else: 
            return val
    return aux

def w2(f,g,dt,tol=0.0, debug=False):
    u = w2_peval(f,dt,tol)
    return u(g,debug=debug)


    
