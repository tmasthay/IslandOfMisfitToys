import torch
from elastic_custom import *
import matplotlib.pyplot as plt

def case1():
    n = 2000
    num_prob = 100
    nex = 2
    x = torch.linspace(-10,10,n)
    f = torch.exp(-x**2).repeat(nex,1)
    F = torch.cumsum(f,dim=1)
    F = F / F[:,-1].unsqueeze(1)
    p = torch.linspace(0.0, 1.0, num_prob).repeat(nex,1)

    Q = my_quantile2(F, x.repeat(nex,1), p)

    for ex in range(nex):
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10,7))

        axs[0].plot(x, f[ex], label='pdf')
        axs[0].plot(x, F[ex], label='cdf')
        axs[0].legend()

        axs[1].plot(p[ex], Q[ex], label='Q')
        axs[1].legend()

        plt.savefig(f'tmp{ex}.jpg')
        plt.clf()

def case2():
    n = 2000
    num_prob = 1000
    nex = 2
    x = torch.linspace(-10,10,n)
    f = torch.exp(-x**2).repeat(nex, 1)
    g = torch.exp(-x**2).repeat(nex,1)
    def renorm(h):
       assert( len(h.shape) == 2 )
       u = h**2
       return u / torch.sum(u, dim=1).unsqueeze(1) 
    w2g = w2_peval(g, x=x, renorm=renorm, num_prob=num_prob)
    res = w2g(f)
    print(res)

if( __name__ == "__main__" ):
    case2()
