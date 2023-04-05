import numpy as np
from numpy import linalg as LA
#def func1():
#    print('func1')


def W1_metric(f, g, t, Nt, dt)
# normalization
    f = f / (LA.norm(f,1)*dt)
    g = g / (LA.norm(g,1)*dt)
# numerical integration
    F = np.zeros(Nt)
    G = np.zeros(Nt)
    F[0] = f[0]
    G[0] = g[0]
    for i in range(1, Nt):
        F[i] = F[i-1] + f[i]
        G[i] = G[i-1] + g[i]
    F = F * dt
    G = G * dt
# inverse of G 
    G_inv = np.zeros(Nt)
    for i in range(0, Nt):
        y = F[i]
        ind_g = np.where( y >= G)
        if size(ind_g) == 0
            G_inv[i] = t[end]
        else
            G_inv[i] = t[ind_g[0]]
    w1 = sum(abs(t-G_inv) * f * dt)
    return w1
