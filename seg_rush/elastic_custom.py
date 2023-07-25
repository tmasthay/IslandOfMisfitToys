import torch
import matplotlib.pyplot as plt
import deepwave
from deepwave import elastic
from custom_losses import *
from deepwave_helpers import *
import numpy as np

global device
device = torch.device('cuda' if torch.cuda.is_available()
                      else 'cpu')

def get_data(**kw):
    #import global vars
    global device

    #initialize defaults
    d = {
        'nx': 500,
        'ny': 500,
        'dx': 4.0,
        'dy': 4.0,
        'n_shots': 1,
        'first_source': 0,
        'source_depth': 2,
        'd_source': 1,
        'first_receiver': 0,
        'receiver_depth': 2,
        'd_receiver': 1,
        'freq': 1.0,
        'nt': 800,
        'dt': 0.001
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
        2 + torch.arange(d['ny']-3),
        2 + torch.arange(d['nx']-3)
    )
    d['grid_y'] = d['grid_y'].to(device)
    d['grid_x'] = d['grid_x'].to(device)

    #set source location
    d['src_loc'] = torch.stack((d['grid_x'], d['grid_y']), dim=2)
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

    def forward():
        def helper(**kw):
            def amp_helper(s):
                src_amps = d[s](d['dy']*d['grid_y'],d['dx']*d['grid_x'], **kw) \
                    .reshape(-1)
                return (src_amps.view(d['n_shots'],*src_amps.shape,1) \
                    * d['wavelet'].view(d['n_shots'],1,-1)
                ) \
                .to(device)
                
            src_amps_y = amp_helper('force_y')
            src_amps_x = amp_helper('force_x')  
                
            return elastic(*d['lamb_mu_buoy'],
                d['dx'],
                d['dt'],
                source_amplitudes_y=src_amps_y,
                source_amplitudes_x=src_amps_x,
                source_locations_y=d['src_loc'],
                source_locations_x=d['src_loc'],
                receiver_locations_y=d['src_loc'],
                receiver_locations_x=d['src_loc'],
                pml_freq=d['freq']
            )[-2]
        return helper
    
    d['forward'] = forward() 
    u = {**d, **kw}
    return u

def forward_samples(sy, sx, d, **kw):
    defaults = {
        'sigy': 0.1,
        'sigx': 0.1
    }
    curr_kw = {**defaults, **kw}
    
    res = torch.empty(len(sy), 
        len(sx), 
        d['n_shots'],
        d['n_receivers_per_shot'],
        d['nt']
    )
    for (i,yy) in enumerate(sy):
        for (j,xx) in enumerate(sx):
            d['forward'](muy=yy, mux=xx, **curr_kw)
            print('Finished!')
            exit(-1)
     

def go():
    #declare globals
    global device

    d = get_data()
    num_samples_y = 50
    num_samples_x = 50

    y_min = d['grid_y'][d['first_source']][0] + 1
    y_max = d['grid_y'][-1][0] - 1
   
    x_min = d['grid_x'][0][0] + 1
    x_max = d['grid_x'][0][-1] - 1

    sy = torch.linspace(y_min, y_max, num_samples_y).to(device)
    sx = torch.linspace(x_min, x_max, num_samples_x).to(device)

    forward_samples(sy,sx,d)

if( __name__ == "__main__" ):
    go()
