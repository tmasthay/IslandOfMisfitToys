import torch
import matplotlib.pyplot as plt
import deepwave
from deepwave import elastic
from custom_losses import *
from deepwave_helpers import *
import numpy as np
import argparse
from random import randint
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

global device
global cmap
device = torch.device('cuda:1')

plt.rcParams['text.usetex'] = True

def vertical_stratify(ny, nx, layers, values):
    global device
    assert( len(layers) == len(values) - 1 )
    assert( len(values) >= 1 )
    u = values[0] * torch.ones(ny,nx, device=device)
    if( len(layers) > 0 ):
        for l in range(len(layers)-1):
            u[layers[l]:layers[l+1], :] = values[l+1]
        u[layers[-1]:] = values[-1]
    return u

def uniform_vertical_stratify(ny, nx, values):
    layers = [i*ny // len(values) for i in range(1,len(values))]
    return vertical_stratify(ny, nx, layers, values)

def plot_material_params(vp, vs, rho):
    global cmap

    up = vp.cpu().detach()
    us = vs.cpu().detach()
    urho = rho.cpu().detach()
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(10,7))
    axs[0].imshow(up, cmap=cmap)
    axs[1].imshow(us, cmap=cmap)
    axs[2].imshow(urho, cmap=cmap)

    plt.savefig('params.jpg')
    plt.clf()
         
def get_data(**kw):
    #import global vars
    global device

    #initialize defaults
    d = {
        'nx': 250,
        'ny': 250,
        'dx': 4.0,
        'dy': 4.0,
        'n_shots': 1,
        'first_source': 1,
        'source_depth': 2,
        'd_source': 1,
        'first_receiver': 0,
        'receiver_depth': 2,
        'd_receiver': 1,
        'freq': 10.0,
        'nt': 1600,
        'dt': 0.001,
        'ofs': 1
    }

    #set material parameters
#    vp_background = 1500.0
#    vs_background = 1100.0
#    rho_background = 2200.0
#    d['vp'] = vp_background * torch.ones(d['ny'], d['nx'], device=device)
#    d['vs'] = vs_background * torch.ones(d['ny'], d['nx'], device=device)
#    d['rho'] = rho_background * torch.ones(d['ny'], d['nx'], device=device)  

    ufs = lambda v : uniform_vertical_stratify(d['ny'], d['nx'], v)
    d['vp'] = ufs([1500.0, 1700.0, 1900.0, 2100.0, 3000.0, 2000.0, 1800.0, 1100.0])
    d['vs'] = ufs([1100.0, 1300.0, 1500.0, 1900.0, 2800.0, 1800.0, 1100.0, 900.0])
    d['rho'] = ufs([2200.0])

    plot_material_params(d['vp'], d['vs'], d['rho'])

    #set per shot numbers
    d['n_sources_per_shot'] = (d['nx']-1)*(d['ny']-1)
    d['n_receivers_per_shot'] = (d['nx']-1)

    #set ricker peak time
    d['peak_time'] = 1.5 / d['freq']
    d['wavelet'] = deepwave.wavelets.ricker(d['freq'],
        d['nt'],
        d['dt'],
        d['peak_time']
    ) \
    .to(device)
 
     
    #set source info
#    d['grid_y'], d['grid_x'] = torch.meshgrid(torch.arange(d['ny']-1),
#        torch.arange(d['nx']-1)
#    )
#    d['grid_y'] = d['grid_y'].to(device)
#    d['grid_x'] = d['grid_x'].to(device)
    d['grid_y'], d['grid_x'] = torch.meshgrid(
        d['ofs'] + torch.arange(d['ny']-d['ofs']-1),
        d['ofs'] + torch.arange(d['nx']-d['ofs']-1)
    )
    d['grid_y'] = d['grid_y'].to(device)
    d['grid_x'] = d['grid_x'].to(device)

    #set source location
    d['src_loc'] = torch.stack((d['grid_y'], d['grid_x']), dim=2)
    d['rec_loc'] = d['src_loc'][d['receiver_depth']].unsqueeze(0).to(device)
    d['src_loc'] = d['src_loc'].view(1,-1,d['src_loc'].shape[-1]).to(device)
   
    def my_force_y(y,x, **kwargs):
        local_amp = kwargs.get('amp', 1e9)
        local_mux = kwargs.get('mux', d['nx'] / 2.0 * d['dx'])
        local_muy = kwargs.get('muy', d['ny'] / 2.0 * d['dy'])
        local_sigx = kwargs.get('sigx', d['dx'])
        local_sigy = kwargs.get('sigy', d['dy'])
        exponent1 = -((x-local_mux)**2 /  (2*local_sigx**2))
        exponent2 = -((y-local_muy)**2 / (2*local_sigy**2))
        return local_amp * torch.exp(exponent1 + exponent2)

    def my_force_x(y,x, **kwargs):
        local_amp = kwargs.get('amp', 1e9)
        local_mux = kwargs.get('mux', d['nx'] / 2.0 * d['dx'])
        local_muy = kwargs.get('muy', d['ny'] / 2.0 * d['dy'])
        local_sigx = kwargs.get('sigx', d['dx'])
        local_sigy = kwargs.get('sigy', d['dy'])
        exponent1 = -((x-local_mux)**2 /  (2*local_sigx**2))
        exponent2 = -((y-local_muy)**2 / (2*local_sigy**2))
        return local_amp * torch.exp(exponent1 + exponent2)

    d['force_y'] = my_force_y
    d['force_x'] = my_force_x

    d['lamb_mu_buoy'] = deepwave.common.vpvsrho_to_lambmubuoyancy(d['vp'],
        d['vs'],
        d['rho']
    )
    d['vp'] = d['vp'].to('cpu')
    d['vs'] = d['vs'].to('cpu')
    d['rho'] = d['rho'].to('cpu')
    del d['vp']
    del d['vs']
    del d['rho']
    torch.cuda.empty_cache()

    def amp_helper(s, **kw):
        src_amps = d[s](d['dy']*d['grid_y'],d['dx']*d['grid_x'], **kw) \
            .reshape(-1)
        result = (src_amps.view(d['n_shots'],*src_amps.shape,1) \
            * d['wavelet'].view(d['n_shots'],1,-1)
        ) \
        .to(device)
        return result

    def forward():
        def helper(**kw):        
            src_amps_y = amp_helper('force_y', **kw)
            src_amps_x = amp_helper('force_x', **kw)
                
            return elastic(*d['lamb_mu_buoy'],
                d['dx'],
                d['dt'],
                source_amplitudes_y=src_amps_y,
                source_locations_y=d['src_loc'],
                receiver_locations_y=d['src_loc'],
                pml_freq=d['freq']
            )[-2]
        return helper

    d['amp_helper'] = amp_helper 
    d['forward'] = forward() 
    u = {**d, **kw}
    return u

def gifify(u, step_size, name, remove_jpg=True, renormalize=True):
    global cmap
    name = name.replace('.gif', '')
    kw = {}
    if( renormalize ):
        kw['vmin'] = u.max()
        kw['vmax'] = u.min()
    for k in range(0,u.shape[-1],step_size):
        plt.imshow(u[:,:,k], cmap=cmap, aspect='auto', **kw)
        plt.title(f'Step {k}')
        plt.colorbar()
        plt.savefig(f'tmp{k}.jpg')
        plt.clf()
    print(f'Generating {name}.gif')
    os.system(f'convert -delay 20 -loop 0 $(ls -tr tmp*.jpg) {name}.gif')
    if( remove_jpg ):
        print('Removing tmp*.jpg')
        os.system('rm tmp*.jpg')

def forward_samples(sy, sx, d, **kw):
    defaults = {
        'sigy': d['dy'],
        'sigx': d['dx'],
        'forward': 'forward.pt',
        'full': 'full'
    }
    curr_kw = {**defaults, **kw}
   
    l = d['ny'] - 2*d['ofs'] 
    m = d['nx'] - 2*d['ofs'] 
    res = torch.empty(len(sy), 
        len(sx), 
        d['n_shots'],
        l,
        d['nt']
    )
    norm_image = False
    p_y_idx = randint(2, len(sy)-2)
    p_x_idx = randint(2, len(sx)-2)
    for (i,yy) in enumerate(sy):
        for (j,xx) in enumerate(sx):
            full_data = d['forward'](muy=yy, mux=xx, **curr_kw)
            if( i == p_y_idx and j == p_x_idx ):
                tmp = full_data[0,:,:].reshape(l,m,d['nt']).cpu().detach()
                gifify(tmp, 100, curr_kw['full'], True, norm_image)
            res[i,j,0,:,:] = full_data[:d['n_shots']][0,:l,:]
            print(f'{i},{j}')
    make_receiver_plot = True
    if( d['n_shots'] == 1 and make_receiver_plot ):
        if( len(sy) * len(sx) < 200 ):
            res_reshaped = res.reshape(len(sy)*len(sx), l, d['nt'])
            res_reshaped = res_reshaped.permute(1,2,0).cpu()
            print('Making gif of receiver data')
            gifify(res_reshaped, 1, 'receiver', True, norm_image)
    print('Saving data')
    torch.save(res, curr_kw['forward'])
    return res

def landscape(res, loss, name, **kw):
    global cmap
    ref_idx = kw.get('ref_idx', [res.shape[0] // 2, res.shape[1] // 2])
    u = res[ref_idx[0], ref_idx[1], 0, :, :] 
    losses = torch.empty(res.shape[0], res.shape[1])
    for i in range(res.shape[0]):
        for j in range(res.shape[1]):
            losses[i,j] = loss(u, res[i,j,0,:,:])
    plt.imshow(losses, cmap=cmap)
    plt.colorbar()
    plt.savefig(name)

def landscape_peval(res, loss_builder, name, **kw):
    global cmap
    ref_idx = kw.get('ref_idx', [res.shape[0] // 2, res.shape[1] // 2])
    u = res[ref_idx[0], ref_idx[1], 0, :, :]
    losses = torch.empty(res.shape[0], res.shape[1])
    loss_peval = loss_builder(u, **kw)
    for i in range(res.shape[0]):
        for j in range(res.shape[1]):
             losses[i,j] = loss_peval(res[i,j,0,:,:])
    plt.imshow(losses, cmap=cmap)
    plt.colorbar()
    plt.savefig(name) 

def slice2(u, idx):
    return u[torch.arange(u.shape[0]).unsqueeze(1), idx]

def frac_ss(x, y, tau=1e-16):
    idx = torch.clamp(torch.searchsorted(x,y), max=x.shape[-1]-1)
    idx_left = torch.clamp(idx-1, min=0)
    val_right = slice2(x, idx)
    val_left = slice2(x, idx_left)
    dval = torch.max(tau * torch.ones(y.shape).to(val_right.device),
        val_right - val_left
    )
    alpha = (val_right - y) / dval
    return alpha * idx_left + (1.0 - alpha) * idx

def frac_idx(x, idx):
    right_idx = torch.ceil(idx).to(torch.long)
    left_idx = torch.floor(idx).to(torch.long)
    alpha = idx - torch.floor(idx)
    x_right = slice2(x, right_idx)
    x_left = slice2(x, left_idx)
    return alpha * x_right + (1.0 - alpha) * x_left

def my_quantile(cdf, x, p, tau=1e-16):
    #cdf -- F(x) for density f supported on x
    #x -- domain of sample space
    #p -- quantiles we want to evaluate
    idx = torch.clamp(torch.searchsorted(cdf, p), max=cdf.shape[-1]-1)
    idx_left = torch.clamp(idx-1, min=0)
    cdf_right = slice2(cdf, idx)
    cdf_left = slice2(cdf, idx_left)
    dcdf = torch.max(tau * torch.ones(p.shape).to(cdf_right.device),
        cdf_right - cdf_left
    )
    alpha = (cdf_right - p) / dcdf
    x_left = slice2(x, idx_left)
    x_right = slice2(x, idx)
    return alpha * x_left + (1.0 - alpha) * x_right

def my_quantile2(cdf, x, p, tau=1e-16):
    idx = frac_ss(cdf, p, tau)
    return frac_idx(x, idx)
   
def w2_peval(g, **kw): 
    global device
    assert( len(g.shape) == 2 )

    x = kw['x'].to(device)
    dx = x[1] - x[0]
    num_prob = kw['num_prob']
    renorm = kw['renorm']
    x = x.repeat(g.shape[0],1)
    g = renorm(g).to(device)
    p = torch.linspace(0.0, 1.0, num_prob).repeat(g.shape[0],1).to(device)
    G = torch.cumsum(g, dim=1)
    G = G / G[:,-1].unsqueeze(1)
    Q = my_quantile2(G, x, p)
    for (i,qq) in enumerate(Q):
         fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(11,8))
         ax[0].plot(x[i].cpu(), g[i].cpu(), label=f'$g_{i}(x)$')
         ax[0].plot(x[i].cpu(), G[i].cpu(), label=f'$G_{i}(x)$')
         ax[1].plot(p[i].cpu(), qq, label=f'$Q(p)$')
         ax[0].legend()
         ax[1].legend()
         plt.savefig(f'q{i}.jpg')
         plt.clf()
    def helper(f):
        f = renorm(f).to(device)
        F = torch.cumsum(f, dim=1)
        F = F / F[:,-1].unsqueeze(1)
        transport = frac_idx(x, frac_ss(Q, F)) - x
        return torch.sum(transport**2 * f) * dx
    return helper

def go():
    #declare globals
    global device
    global cmap

    parser = argparse.ArgumentParser()
    parser.add_argument('--nsy', type=int, default=10)
    parser.add_argument('--nsx', type=int, default=10)
    parser.add_argument('--norun', action='store_true')
    parser.add_argument('--misfit', type=str, default='w2')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--cmap', type=str, default='seismic')
    parser.add_argument('--forward', type=str, default='forward.pt')
    parser.add_argument('--full', type=str, default='full')
    args = parser.parse_args()
    device = torch.device(f'cuda:{args.gpu}')
    cmap = args.cmap
 
    d = get_data()

    ofs = 30
    y_min = (d['grid_y'][d['first_source']][0] + ofs) * d['dy']
    y_max = (d['grid_y'][-1][0] - ofs) * d['dy']
   
    x_min = (d['grid_x'][0][0] + ofs) * d['dx']
    x_max = (d['grid_x'][0][-1] - ofs) * d['dx']

    sy = torch.linspace(y_min, y_max, args.nsy).to(device)
    sx = torch.linspace(x_min, x_max, args.nsx).to(device)

    if( not args.norun ):
        forward_samples(sy,sx,d)
    
    res = torch.load('forward.pt').to(device)
    if( args.misfit.lower() == 'l2' ):
        def loss(x,y):
            dom = max([float(torch.norm(x)**2), float(torch.norm(y)**2)])
            return torch.norm(x-y)**2 / dom
        landscape(res, loss, 'L2.jpg')
    elif( args.misfit.lower() == 'w2' ):
        t = torch.linspace(0.0, (d['nt']-1)*d['dt'], d['nt'])
        def renorm(f):
            assert( len(f.shape) == 2 )
            u = f**2
            return u / torch.sum(u,dim=1).unsqueeze(1)
        landscape_peval(res, 
            w2_peval, 
            'W2.jpg', 
            x=t.to(device),
            renorm=renorm, 
            num_prob=100
        )

if( __name__ == "__main__" ):
    go()
