# smart_quantile_cython.pyx
# cython: boundscheck=False
# cython: cdivision=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: language_level=3

import numpy as np
cimport numpy as cnp
from scipy.integrate import cumulative_trapezoid

def get_flat_subintervals(cnp.ndarray[float, ndim=1] x, double tol=0.0):
    cdef int i, start, prev, runner, curr
    cdef cnp.ndarray[long, ndim=1] idx = np.where(x <= tol)[0]
    cdef list flat_int = []

    if len(idx) > 1:
        start = idx[0]
        prev = idx[0]
        runner = 1
        for i in range(1, len(idx)):
            curr = idx[i]
            if curr == prev + 1:
                runner += 1
            else:
                if runner > 2:
                    flat_int.append((start+1, prev-1))
                start = curr
            prev = curr

        if len(flat_int) > 0:
            return flat_int
        else:
            return [(np.inf, np.inf)]
    else:
        return [(np.inf, np.inf)]

def smart_quantile(
        cnp.ndarray[float, ndim=1] x,
        cnp.ndarray[float, ndim=1] pdf, 
        cnp.ndarray[float, ndim=1] cdf, 
        cnp.ndarray[float, ndim=1] p, 
        double tol=0.0
):
    cdef list flat_int = get_flat_subintervals(pdf)
    cdef int sidx, N, P, i, i_x
    cdef double delta, alpha

    N = len(x)
    P = len(p)
    cdef cnp.ndarray[double, ndim=1] q = np.empty(P)
    q[0] = x[0]
    q[len(q)-1] = x[len(x)-1]

    i = 1
    i_x = 0
    flat_int = [(np.inf, np.inf)]
    sidx = 0

    for i in range(1, P - 1):
        if i_x == N - 1:
            q[i:] = x[1]
            break

        while( cdf[i_x] > p[i] or p[i] > cdf[i_x + 1] ):
            i_x += 1
            csub = flat_int[sidx]
            if csub[0] <= i_x and i_x <= csub[1]:
                i_x = csub[1] + 1
                if sidx < len(flat_int) - 1:
                    sidx += 1

            if i_x == N - 1:
                q[i:] = x[1]
                break

        if i_x == N - 1:
            q[i:] = x[1]
            break

        delta = cdf[i_x + 1] - cdf[i_x]
        if delta > 0:
            alpha = (p[i] - cdf[i_x]) / delta
            q[i] = (1.0 - alpha) * x[i_x] + alpha * x[i_x + 1]
        else:
            q[i] = x[i_x]

    return q

def smart_quantile_peval(
        g,
        x,
        tol=0.0, 
        restrict=None,
        explicit_restrict=None
    ):
    dx = x[1] - x[0]
    x = np.array(x, dtype=np.float32)
    G = np.array(cumulative_trapezoid(g, dx=dx, initial=0.0), dtype=np.float32)
    def helper(p, tau=tol):
        return smart_quantile(x, g, G, p, tol)
    return helper


