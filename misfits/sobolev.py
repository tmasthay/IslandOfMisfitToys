import numpy as np

def sobolev_norm(f, s=0, **kw):
    """
    Compute the Sobolev norm of the difference between two 1D functions f(x) and g(x)
    over the domain specified by the array `domain`, up to order k.
    
    Parameters
    ----------
    f : function
        any function in $H^{s}$.
    s : int
        Sobolev space smoothness parameter.
        Defaults to 0.
    kw : kw arguments
        ot : float
            "origin in time", i.e., "$t_0$".
        dt : float
            time sampling interval
        nt : int
            number of time samples
        'sample' : 3-tuple
            (ot, dt, nt) alternative input format
    Returns
    -------
    float
        $\|f\|_{H^{s}}$
    """
    if( 'sample' in kw.keys() ):
        ot = kw['sample'][0]
        dt = kw['sample'][1]
        nt = kw['sample'][2]
    else:
        ot = kw['ot']
        dt = kw['dt']
        nt = kw['nt']
    xi = np.fft.fftfreq(nt, d=dt)
    f_hat = np.exp(-2j * np.pi * ot * xi) * np.fft.fft(f) * dt

    xi = np.fft.fftshift(xi)
    f_hat = np.fft.fftshift(f_hat)
    g = (1 + np.abs(xi)**2)**s * np.abs(f_hat)**2
    dxi = xi[1] - xi[0]
    res = np.trapz(g, dx=dxi)
    return res, xi, g

