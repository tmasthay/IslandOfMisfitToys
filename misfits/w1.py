import numpy as np
#def func1():
#    print('func1')


def W1_metric(f, g, Nt): 
#CDF
    F = np.zeros(Nt)
    G = np.zeros(Nt)
    for i in range(0, Nt):
        F[i] = np.sum(f[0:i])
        G[i] = np.sum(g[0:i])
# inverse
    w1 = np.sum(np.abs(F-G))
    return w1
