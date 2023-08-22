import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
import deepwave
from deepwave import elastic
import numpy as np
import argparse
from random import randint
from warnings import warn
from typing import Annotated as Ant
from abc import ABC, abstractmethod

from .custom_losses import *
from .deepwave_helpers import *

global device
global cmap
device = torch.device('cuda:0')

plt.rc('text', usetex=False)

class Data:
    __slots__ = [
        'devices',

        'nx', 
        'ny',
        'nt', 
        'dx', 
        'dy',
        'dt',

        'vp', 
        'vs', 
        'rho', 
         
        'n_shots', 

        'fst_src', 
        'src_depth', 
        'd_src',
        'src_loc', 
        'n_src_per_shot',

        'fst_rec', 
        'rec_depth', 
        'd_rec',
        'n_rec_per_shot', 
        'rec_loc', 

        'freq',  
        'wavelet',
        
        'ofs', 
    ]
    devices: Ant[list, 'List of devices']

    vp: Ant[torch.Tensor, 'P-wave velocity', '0.0']
    vs: Ant[torch.Tensor, 'S-wave velocity', '0.0', 'vp']
    rho: Ant[torch.Tensor, 'Density', '0.0@s']

    n_shots: Ant[int, 'Number of shots']

    fst_src: Ant[int, 'First source index (x-dir)', '1', 'nx-1']
    src_depth: Ant[int, 'Source depth index (y-dir)', '1', 'ny-1']
    d_src: Ant[int, 'Index delta of sources (x-dir)', '1', 'nx-2']
    n_src_per_shot: Ant[int, 'Num sources per shot', '1']
    src_loc: Ant[torch.Tensor, 'Source locations']

    fst_rec: Ant[int, 'First receiver index (x-dir)', '1', 'nx-1']
    rec_depth: Ant[int, 'Rec. depth index (y-dir)', '1', 'ny-1']
    d_rec: Ant[int, 'Index delta of receivers (x-dir)', '1', 'nx-2']
    n_rec_per_shot: Ant[int, 'Num receivers per shot', '1']
    rec_loc: Ant[torch.Tensor, 'Receiver locations']

    nx: Ant[int, 'Number of horizontal dofs', '1']
    ny: Ant[int, 'Number of vertical dofs', '1']
    nt: Ant[int, 'Number of time steps', '1']
    dx: Ant[float, 'Grid size horizontal', '0.0s']
    dy: Ant[float, 'Grid size vertical', '0.0s']
    dt: Ant[float, 'Time step', '0.0#s']

    freq: Ant[float, 'Characteristic Ricker frequency', '0.0#s']
    wavelet: Ant[torch.Tensor, 'Characteristic source time signature']

    ofs: Ant[int, 'Padding for src_loc landscape', '1', 'min(nx-1,ny-1)']

    def __init__(self, **kw):
        self.devices = [torch.device('cuda:%d'%i) \
            for i in range(torch.cuda.device_count())] + [torch.device('cpu')]
        self.vp = read_tensor(kw['vp'], self.devices[0])
        self.vs = read_tensor(kw['vs'], self.devices[0])
        self.rho = read_tensor(kw['rho'], self.devices[0])

        assert len(self.vp.shape) == 2, 'vp dimension mismatch, ' \
            f'Expected 2 dimensions but got {len(self.vp.shape)} in the ' \
            f'shape of {self.vp.shape}'

        assert self.vp.shape == self.vs.shape \
            and self.vs.shape == self.rho.shape, \
            'vp,vs,rho dimension mismatch, ' \
                f'{self.vp.shape},{self.vs.shape},{self.rho.shape}'

        #get shot info
        self.n_shots = kw['n_shots']

        #get source info
        self.fst_src = kw['fst_src']
        self.n_src_per_shot = kw['n_src_per_shot']
        self.src_depth = kw['src_depth']
        self.d_src = kw['d_src']
        self.src_loc = kw['src_loc']
    
        #get receiver info
        self.fst_rec = kw['fst_rec']
        self.n_rec_per_shot = kw['n_rec_per_shot']
        self.rec_depth = kw['rec_depth']
        self.d_rec = kw['d_rec']
        self.rec_loc = kw['rec_loc']

        #get grid info
        self.ny, self.nx, self.nt = *self.vp.shape, kw['nt']
        self.dy, self.dx, self.dt = kw['dy'], kw['dx'], kw['dt']

        #get time signature info
        self.freq = kw['freq']
        self.wavelet = deepwave.wavelets.ricker(kw['freq'], 
            kw['nt'], 
            kw['dt'],
            kw['peak_time']
        )
    
        #get offset info for optimization landscape plots
        self.ofs = kw.get('ofs', 1)\

    def update(self, **kw):
        for k,v in kw.items():
            if( k in self.__slots__ ):
                setattr(self, k, v)
            else:
                warn(f'Attribute {k} does not exist...skipping ')

class DataGenerator(Data, ABC):
    __slots__ = [
        'custom'
    ]
    custom: Ant[dict, 'Custom parameters for user flexibility']
    def __init__(self, **kw):
        super().__init__(**kw)
        self.custom = dict()
        new_keys = set(kw.keys()).difference(set(super().__slots__))
        for k in new_keys:
            self.custom[k] = kw[k]

    @abstractmethod
    def force(y,x,comp,**kw):
        pass

    @abstractmethod
    def forward(**kw):
        pass


def get_data(**kw):
    #import global vars
    global device

    #initialize defaults
    d = {
        'nx': 250, #int
        'ny': 250, #int
        'dx': 4.0, #float
        'dy': 4.0, #float
        'n_shots': 1, #int
        'first_source': 1, #int
        'source_depth': 2, #int 
        'd_source': 1, #int
        'first_receiver': 0, #int
        'receiver_depth': 2, #int
        'd_receiver': 1, #int
        'freq': 10.0,
        'nt': 1600,
        'dt': 0.001,
        'ofs': 1
    }  

    ufs = lambda v : uniform_vertical_stratify(d['ny'], d['nx'], v, device)
    d['vp'] = ufs([1500.0, 1700.0, 3000.0, 1000.0, 600.0, 3000.0])
    d['vs'] = ufs([1100.0, 1300.0, 1500.0, 800.0, 500.0, 2700.0])
    d['rho'] = ufs([2200.0])

    # plot_material_params(d['vp'], d['vs'], d['rho'])

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
    make_receiver_plot = False
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
    defaults = {'shuffle': 1, 'title': ''}
    kw = {**defaults, **kw}
    ref_idx = kw.get('ref_idx', [res.shape[0] // 2, res.shape[1] // 2])
    if( 'ref' in kw.keys() ):
        u = kw['ref']
    else:
        u = res[ref_idx[0], ref_idx[1], 0, :, :] 
    losses = torch.empty(res.shape[0], res.shape[1])
    for i in range(res.shape[0]):
        for j in range(res.shape[1]):
            curr = res[i,j,0,:,:]
            if( kw['snr'] > 0.0 ):
                sigma = torch.sqrt((curr**2).mean()) / kw['snr']
                curr += torch.randn_like(curr) * sigma
            losses[i,j] = loss(u, curr)
    if( kw['shuffle'] == 1 ): 
        plt.imshow(losses, cmap=cmap)
        plt.colorbar()
        plt.title(kw['title'])
        plt.xlabel('Source horizontal location (km)')
        plt.ylabel('Source depth (km)')
        plt.savefig(name)
        plt.clf()
    return losses

def landscape_peval(res, loss_builder, name, **kw):
    global cmap
    defaults = {'shuffle': 1}
    kw = {**defaults, **kw}
    ref_idx = kw.get('ref_idx', [res.shape[0] // 2, res.shape[1] // 2])
    if( 'ref' in kw.keys() ):
        u = kw['ref']
    else:
        u = res[ref_idx[0], ref_idx[1], 0, :, :]
    losses = torch.empty(res.shape[0], res.shape[1])
    loss_peval = loss_builder(u, **kw)
    for i in range(res.shape[0]):
        for j in range(res.shape[1]):
             losses[i,j] = loss_peval(res[i,j,0,:,:])
    if( kw['shuffle'] == 1 ):
        plt.imshow(losses, cmap=cmap)
        plt.colorbar()
        plt.title(kw['title'])
        plt.xlabel('Source horizontal location (km)')
        plt.ylabel('Source depth (km)')
        plt.savefig(name)
        plt.clf()
    return losses

def slice2(u, idx):
    if( idx.min() < 0 ):
        print(idx.shape)
        print(u.shape)
        assert idx.min() >= 0, f'idx.min() == {idx.min()} < 0'
    elif( idx.max() >= u.shape[1] ):
        print(idx.shape)
        print(u.shape)
        #assert idx.max() < u.shape[0], f'idx.max() == {idx.max()} >= {u.shape[0]}'
    return u[torch.arange(u.shape[0]).unsqueeze(1), idx]

def frac_ss(x, y, tau=1e-33, left_right=False):
    if( left_right ):
        idx = torch.clamp(torch.searchsorted(x,y,right=True), 
            max=x.shape[-1]-1
        )
        idx_left = torch.clamp(torch.searchsorted(x,y), min=0)
    else: 
        idx = torch.clamp(torch.searchsorted(x,y), max=x.shape[-1]-1)
        idx_left = torch.clamp(idx-1, min=0)
    val_right = slice2(x, idx)
    val_left = slice2(x, idx_left)
    dval = val_right - val_left
    dval = torch.where(dval != 0, dval, torch.ones_like(dval))
    alpha = (val_right - y) / dval
    return torch.clamp( alpha * idx_left + (1.0 - alpha) * idx, min=0, max=x.shape[-1]-1 )

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

    dcdf = cdf_right - cdf_left
    dcdf = torch.where(dcdf != 0, dcdf, torch.ones_like(dcdf))
    alpha = (cdf_right - p) / dcdf
    x_left = slice2(x, idx_left)
    x_right = slice2(x, idx)
    return alpha * x_left + (1.0 - alpha) * x_right

def my_quantile2(cdf, x, p, tau=1e-33):
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
    #Q = my_quantile2(G, x, p)
    num_plots = 0
    def helper(f):
        f = renorm(f).to(device)
        F = torch.cumsum(f, dim=1)
        F = F / F[:,-1].unsqueeze(1)
        Q = my_quantile2(G, x, F)
        integrand = (Q-x)**2 * f
#        for i in range(Q.shape[0]):
#            fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(11,8))
#            ax[0,0].plot(x[i].cpu(), f[i].cpu(), label=f'pdf{i}')
#            ax[0,1].plot(x[i].cpu(), F[i].cpu(), label=f'cdf{i}')
#            ax[0,0].plot(x[i].cpu(), g[i].cpu(), label=f'pdf_ref{i}')
#            ax[0,1].plot(x[i].cpu(), G[i].cpu(), label=f'cdf_ref{i}')
#            ax[1,0].plot(x[i].cpu(), Q[i].cpu(), label=f'transport{i}')
#            ax[1,0].plot(x[i].cpu(), x[i].cpu(), label=f'identity{i}')
#            ax[1,1].plot(x[i].cpu(), (Q[i].cpu()-x[i].cpu())**2, label=f'iso{i}')
#            ax[1,1].plot(x[i].cpu(), integrand[i].cpu(), label=f'integrand{i}')
#            ax[0,0].legend()
#            ax[0,1].legend()
#            ax[1,0].legend()
#            ax[1,1].legend()
#            plt.savefig(f'transport{i}.jpg')
#            plt.clf()
        return torch.sum(torch.trapz(integrand, x[0], dim=1))
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
    parser.add_argument('--shuffle', type=int, default=1)
    parser.add_argument('--snr', type=float, default=0.0)
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
    
    res_cpu = torch.load(args.forward).to('cpu')
    ref_u = res_cpu[res_cpu.shape[0] // 2, res_cpu.shape[1] // 2,0,:,:].to(device)
    losses = torch.empty(res_cpu.shape[0], res_cpu.shape[1])
    start_idx = 0
    final_idx = res_cpu.shape[0] // args.shuffle

    if( args.misfit.lower() == 'l2' ):
        plot_title = r'$L^2$ Source Location Optimization Landscape'
    elif( args.misfit.lower() == 'w2' ):
        plot_title = r'$W_2$ Source Location Optimization Landscape'
    if( args.snr > 0.0 ):
        plot_title = r'%s, $\sigma=%.2f$'%(plot_title, args.snr)
    gpu_report = gpu_mem_helper()
    idx = 0
    while( start_idx < res_cpu.shape[0] ):
        s = slice(start_idx, final_idx)
        gpu_report()
        res = res_cpu[s].to(device)
        if( args.misfit.lower() == 'l2' ):
            def loss(x,y):
                return torch.norm(x-y)**2 
            curr = landscape(res, loss, 'L2.jpg', shuffle=args.shuffle,
                title=plot_title, snr=args.snr, forward=args.forward, ref=ref_u)
            idx += 1
            losses[s] = curr.cpu()
            print('Finished current shuffle')
            gpu_report()
        elif( args.misfit.lower() == 'w2' ):
            t = torch.linspace(0.0, (d['nt']-1)*d['dt'], d['nt'])
            def renorm(f):
                assert( len(f.shape) == 2 )
                u = f**2
                return u / torch.sum(u,dim=1).unsqueeze(1)
            curr = landscape_peval(res, 
                w2_peval, 
                'W2.jpg', 
                x=t.to(device),
                renorm=renorm, 
                num_prob=100,
                shuffle=args.shuffle,
                title=plot_title,
                snr=args.snr,
                forward=args.forward,
                ref=ref_u
            )
            losses[s] = curr.cpu()
        torch.cuda.empty_cache()
        start_idx = final_idx
        final_idx = final_idx + res_cpu.shape[0] // args.shuffle
        final_idx = min(res_cpu.shape[0], final_idx)
        if( start_idx == res_cpu.shape[0] ):
            plt.imshow(losses, cmap=cmap)
            plt.colorbar()
            plt.xlabel(r'Source horizontal location (km)')
            plt.ylabel(r'Source depth (km)')
            plt.title(plot_title)
            plt.savefig('%s.jpg'%(args.misfit.lower()))

if( __name__ == '__main__' ):
    go()
    print('FINISHED SUCCESSFULLY')
