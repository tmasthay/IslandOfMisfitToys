import torch
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import deepwave
from deepwave import scalar
from deepwave_helpers import get_file
from torch.cuda.nvtx import range_push, range_pop
import sys
from tqdm import trange

def preprocess_data(**kw):
    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    defaults = {
        'device': device,
        'ny_full': 2301,
        'nx_full': 751,
        'ny': 600,
        'nx': 250,
        'dx': 4.0,
        'file_name': 'marmousi_vp.bin',
        'v_init_lambda': lambda x : \
            torch.tensor(1/gaussian_filter(1/x.numpy(), 40)).to(device),
        'n_shots_full': 115,
        'n_sources_per_shot': 1,
        'd_source': 20,
        'first_source': 10,
        'source_depth': 2,
        'n_receivers_per_shot_full': 384,
        'd_receiver': 6,
        'first_receiver': 0,
        'receiver_depth': 2,
        'freq': 25,
        'nt_full': 750,
        'dt': 0.004,
        'obs_binary': 'marmousi_data.bin',
        'n_shots': 20,
        'n_receivers_per_shot': 100,
        'nt': 300,
        'type': torch.float,
        'training': {
            'n_epochs': 100,
            'prop_profiled': 0.0,
            'stats': dict(),
            'print_freq': 1
        },
        'plotting': {
            'output_binary': 'marmousi_v_inv.bin',
            'output_files': ['example_simple_fwi.jpg'],
            'figsize': (10.5, 10.5),
            'aspect': 'auto',
            'cmap': 'cividis'
        }
    }
    defaults.update({'peak_time': 1.5 / defaults['freq']})
    d = {**defaults, **kw}
    d['training'] = {**defaults['training'], **kw.get('training', {})}
    d['plotting'] = {**defaults['plotting'], **kw.get('plotting', {})}

    d.update({'v_true': \
        torch.from_file(get_file(d['file_name']), 
            size=d['ny_full']*d['nx_full']
        ) \
        .reshape(d['ny_full'], d['nx_full'])})
    
    # Select portion of model for inversion
    d.update({'v_true_downsampled': d['v_true'][:d['ny'], :d['nx']]})

    # Smooth to use as starting model
    d.update({'v_init': d['v_init_lambda'](d['v_true_downsampled'])})
    d.update({'v': d['v_init'].clone()})
    d['v'].requires_grad_()

    d.update({'observed_data': \
            torch.from_file(get_file(d['obs_binary']),
                size=(d['n_shots_full'] \
                    * d['n_receivers_per_shot_full'] \
                    * d['nt_full']
                )
            ) \
            .reshape(d['n_shots_full'], 
                d['n_receivers_per_shot_full'], 
                d['nt_full']
            )
        }
    )
    d['observed_data'] = d['observed_data'][:d['n_shots'], \
            :d['n_receivers_per_shot'], 
            :d['nt']
        ] \
        .to(device)

    # source_locations
    d.update({'source_locations': \
            torch.zeros(d['n_shots'], 
                d['n_sources_per_shot'], 
                2,
                dtype=d['type'], 
                device=device
            )
        }
    )
    d['source_locations'][..., 1] = d['source_depth']
    d['source_locations'][:, 0, 0] = torch.arange(d['n_shots']) \
        * d['d_source'] + d['first_source']

    # receiver_locations
    d.update({'receiver_locations': \
            torch.zeros(d['n_shots'], 
                d['n_receivers_per_shot'],
                2,
                dtype=torch.long, 
                device=device)
        }
    )
    d['receiver_locations'][..., 1] = d['receiver_depth']
    d['receiver_locations'][:, :, 0] = (torch.arange(
        d['n_receivers_per_shot']) * d['d_receiver'] + d['first_receiver']) \
        .repeat(d['n_shots'], 1)

    # source_amplitudes
    d.update({'source_amplitudes': \
            deepwave.wavelets.ricker(d['freq'], d['nt'], d['dt'], 
                d['peak_time']) \
            .repeat(d['n_shots'], d['n_sources_per_shot'], 1) \
            .to(device)
        }
    )

    # Setup optimiser to perform inversion
    d.update({'optimiser': torch.optim.SGD([d['v']], lr=0.1, momentum=0.9)})
    d.update({'loss_fn': torch.nn.MSELoss()})
    d['v_true_downsampled'] = d['v_true_downsampled'].to(device)
    return d

def build_stats(loss, fields):
    d = dict()
    if( 'loss' in fields ): d.update({'loss': loss.item()})
    return d
    
def deploy_training(**d):
    n_epochs = d['training']['n_epochs']
    prof = n_epochs - int(d['training']['prop_profiled'] * n_epochs)
    stats = d['training']['stats']
    print_freq = d['training']['print_freq']

    pbar = trange(n_epochs // print_freq, desc='Training', unit='epoch')

    for epoch in pbar:
        if(epoch == prof): torch.cuda.cudart().cudaProfilerStart()
        if(epoch >= prof): range_push('DL iteration{}'.format(epoch))
        if(epoch >= prof): range_push('Closure')

        def closure():
            d['optimiser'].zero_grad()
            out = scalar(d['v'], 
                d['dx'], 
                d['dt'],
                source_amplitudes=d['source_amplitudes'],
                source_locations=d['source_locations'],
                receiver_locations=d['receiver_locations'],
                pml_freq=d['freq']
            )
            loss = 1e10 * d['loss_fn'](out[-1], d['observed_data'])
            loss.backward()
            torch.nn.utils.clip_grad_value_(
                d['v'],
                torch.quantile(d['v'].grad.detach().abs(), 0.98)
            )
            return loss
        
        if(epoch >= prof): range_pop()
        if(epoch >= prof): range_push('backprop')

        loss = d['optimiser'].step(closure)

        # if( epoch % print_freq == 0 ):
        #     pbar.set_postfix(**build_stats(loss, stats))
        #     pbar.update(1)
        pbar.set_postfix(**build_stats(loss, stats))
        pbar.update(1)
        if(epoch >= prof): range_pop()
        if(epoch >= prof): range_pop()

    if( prof < n_epochs ):
        torch.cuda.cudart().cudaProfilerStop()
    return d

def postprocess(**d):
    p = d['plotting']
    vmin = d['v_true'].min()
    vmax = d['v_true'].max()
    _, ax = plt.subplots(3, figsize=p['figsize'], sharex=True, sharey=True)
    ax[0].imshow(d['v_init'].cpu().T, aspect='auto', cmap=p['cmap'],
                vmin=vmin, vmax=vmax)
    ax[0].set_title("Initial")
    ax[1].imshow(d['v'].detach().cpu().T, aspect='auto', cmap=p['cmap'],
                vmin=vmin, vmax=vmax)
    ax[1].set_title("Out")
    ax[2].imshow(d['v_true'].cpu().T, aspect='auto', cmap=p['cmap'],
                vmin=vmin, vmax=vmax)
    ax[2].set_title("True")
    plt.tight_layout()
    plt.savefig(p['output_files'][0])

    d['v'].detach().cpu().numpy().tofile(p['output_binary'])
