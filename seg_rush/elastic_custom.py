import torch
import matplotlib.pyplot as plt
import deepwave
from deepwave import elastic
from custom_losses import *
from deepwave_helpers import *
import numpy as np
import argparse

global device
device = torch.device('cuda' if torch.cuda.is_available()
                      else 'cpu')

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
        'freq': 1.0,
        'nt': 1600,
        'dt': 0.001,
        'ofs': 1
    }

    #set material parameters
    d['vp'] = 1500 * torch.ones(d['ny'], d['nx'], device=device)
    d['vs'] = 1100 * torch.ones(d['ny'], d['nx'], device=device)
    d['rho'] = 2200 * torch.ones(d['ny'], d['nx'], device=device)

    #set per shot numbers
    d['n_sources_per_shot'] = (d['nx']-1)*(d['ny']-1)
    d['n_receivers_per_shot'] = (d['nx']-1)

    #set ricker peak time
    d['peak_time'] = 0.3 / d['freq']
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
        print(f'amp,muy,mux,sigy,sigx={local_amp},{local_muy},{local_mux},{local_sigy},{local_sigx}')
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

def gifify(u, step_size, name, remove_jpg=True):
    for k in range(0,u.shape[-1],step_size):
        plt.imshow(u[:,:,k], cmap='seismic', aspect='auto')
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
    for (i,yy) in enumerate(sy):
        for (j,xx) in enumerate(sx):
            full_data = d['forward'](muy=yy, mux=xx, **curr_kw)
            res[i,j,0,:,:] = full_data[:d['n_shots']][0,:l,:]
            tmp1 = d['amp_helper']('force_y',
                muy=yy,
                mux=xx,
                **curr_kw
            )
            print(f'{i},{j}')
    make_receiver_plot = False
    if( d['n_shots'] == 1 and make_receiver_plot ):
        res_reshaped = res.reshape(len(sy)*len(sx), l, d['nt'])
        res_reshaped = res_reshaped.permute(1,2,0).cpu()
        input(res_reshaped.shape)
        print('Making gif of receiver data')
        gifify(res_reshaped, 1, 'receiver', True)
    print('Saving data')
    torch.save(res, 'forward.pt')
    return res

def landscape(res, loss, name, **kw):
    ref_idx = kw.get('ref_idx', res.shape[:2] // 2)
    u = res[ref_idx[0], ref_idx[1], 0, :, :] 
    losses = torch.empty(ns, ns)
    for i in range(ns):
        for j in range(ns):
            losses[i,j] = loss(u, v[i,j,0,:,:])
    plt.imshow(losses)
    plt.colorbar()
    plt.savefig(name)
    
def go():
    #declare globals
    global device

    parser = argparse.ArgumentParser()
    parser.add_argument('-nsy', type=int, default=10)
    parser.add_argument('-nsx', type=int, default=10)
    parser.add_argument('-rerun', type=bool, default=True)
    args = parser.parse_args()
 
    d = get_data()

    y_min = d['grid_y'][d['first_source']][0] * d['dy']
    y_max = d['grid_y'][-1][0] * d['dy']
   
    x_min = d['grid_x'][0][0] * d['dx']
    x_max = d['grid_x'][0][-1] * d['dx']

    sy = torch.linspace(y_min, y_max, args.nsy).to(device)
    sx = torch.linspace(x_min, x_max, args.nsx).to(device)

    if( args.rerun ):
        forward_samples(sy,sx,d)
    
    res = torch.load('forward.pt')
    landscape(res, lambda x,y: torch.norm(x-y)**2, 'L2.jpg')

if( __name__ == "__main__" ):
    go()
