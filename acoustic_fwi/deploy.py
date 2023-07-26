import torch
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import deepwave
from deepwave import scalar
from .deepwave_helpers import get_file, make_gif, gpu_mem_helper
from torch.cuda.nvtx import range_push, range_pop
import sys
from tqdm import trange
import matplotlib.gridspec as gridspec
import torchsummary

def preprocess_data(**kw):
    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    gpu_mem = gpu_mem_helper()
    gpu_mem('Before preprocessing')
    defaults = {
        'device': device,
        'ny_full': 2301,
        'nx_full': 751,
        'ny': 600,
        'nx': 250,
        'dx': 4.0,
        'x_start': 0,
        'y_start': 0,
        'shot_start': 0,
        'rec_start': 0,
        'file_name': 'marmousi_vp.bin',
        'v_init_lambda': lambda x : \
            torch.tensor(1/gaussian_filter(1/x.numpy(), 40)),
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
        'loss_fn': torch.nn.MSELoss(),
        'optimiser_lambda': lambda x : \
            torch.optim.SGD([x], lr=0.1, momentum=0.9),
        'loss_fn': torch.nn.MSELoss(),
        'training': {
            'n_epochs': 2,
            'shots_per_batch': 1,
            'prop_profiled': 0.0,
            'stats': {'loss'},
            'print_freq': 1
        },
        'plotting': {
            'output_binary': 'marmousi_v_inv.bin',
            'output_files': ['example_simple_fwi.jpg'],
            'figsize': (10.5, 10.5),
            'aspect': 'auto',
            'cmap': 'gray'
        }
    }
    defaults.update({'peak_time': 1.5 / defaults['freq']})
    d = {**defaults, **kw}
    d['training'] = {**defaults['training'], **kw.get('training', {})}
    d['plotting'] = {**defaults['plotting'], **kw.get('plotting', {})}

    if( type(d['file_name']) == str ):
        d.update({'v_true': \
            torch.from_file(get_file(d['file_name']), 
                size=d['ny_full']*d['nx_full']
            ) \
            .reshape(d['ny_full'], d['nx_full'])})
    else:
        d.update({'v_true': d['file_name']})
    
    #define subsampling slices for physical domain
    y_slice = slice(d['y_start'], d['y_start'] + d['ny'])
    x_slice = slice(d['x_start'], d['x_start'] + d['nx'])

    #define subsampling slices for (source, receiver) domain
    shot_slice = slice(d['shot_start'], 
        d['shot_start'] + d['n_shots']
    )
    rec_slice = slice(d['rec_start'],
        d['rec_start'] + d['n_receivers_per_shot']
    )
    time_slice = slice(0, d['nt'])

    # Select portion of model for inversion
    d.update({'v_true_downsampled': d['v_true'][y_slice, x_slice]})

    # Smooth to use as starting model
    d.update({'v_init': d['v_init_lambda'](d['v_true_downsampled']).to(device)})
    d.update({'v': d['v_init'].clone()})
    d['v'].requires_grad_()

    if( type(d['obs_binary']) == str ):
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
    else:
        d.update({'observed_data': d['obs_binary']})


    d['observed_data'] = d['observed_data'][shot_slice, 
            rec_slice, 
            time_slice
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
    # d.update({'optimiser': torch.optim.SGD([d['v']], lr=0.1, momentum=0.9)})
    d.update({'optimiser': d['optimiser_lambda'](d['v'])})
    d['v_true_downsampled'] = d['v_true_downsampled'].to(device)
    gpu_mem('After preprocessing')
    return d

def build_stats(loss, fields):
    d = dict()
    if( 'loss' in fields ): d.update({'loss': '%.8e'%loss})
    return d
    
def deploy_training(**d):
    param_history = [d['v'].cpu().detach().numpy()]
    loss_history = []
    n_epochs = d['training']['n_epochs']
    shots_per_batch = d['training']['shots_per_batch']
    prof = n_epochs - int(d['training']['prop_profiled'] * n_epochs)
    stats = d['training']['stats']
    print_freq = d['training']['print_freq']

    gpu_mem = gpu_mem_helper()

    pbar = trange(n_epochs // print_freq, desc='Training', unit='epoch')

    for epoch in pbar:
        if(epoch == prof): torch.cuda.cudart().cudaProfilerStart()
        if(epoch >= prof): range_push('DL iteration{}'.format(epoch))
        if(epoch >= prof): range_push('Closure')

        def closure():
            gpu_mem('Closure 0-Entrance')
            epoch_loss = 0.0
            d['optimiser'].zero_grad()
            batch_start = 0
            batch_end = shots_per_batch
            while( batch_start < batch_end ):
                s = slice(batch_start, batch_end)
                gpu_mem('Closure 1-PreForward')
                out = scalar(d['v'], 
                    d['dx'], 
                    d['dt'],
                    source_amplitudes=d['source_amplitudes'][s],
                    source_locations=d['source_locations'][s],
                    receiver_locations=d['receiver_locations'][s],
                    pml_freq=d['freq']
                )
                gpu_mem('Closure 2-PostForward')
                loss = 1e10 * d['loss_fn'](out[-1], d['observed_data'][s])
                epoch_loss += loss.item()
                gpu_mem('Closure 3-PreBackward')
                loss.backward()
                gpu_mem('Closure 4-PostBackward')
                torch.nn.utils.clip_grad_value_(
                    d['v'],
                    torch.quantile(d['v'].grad.detach().abs(), 0.98)
                )
                batch_start = batch_end
                batch_end = min(batch_end + shots_per_batch, d['n_shots'])
                gpu_mem('Closure 5-Exit')
            return epoch_loss
        
        if(epoch >= prof): range_pop()
        if(epoch >= prof): range_push('backprop')

        loss = d['optimiser'].step(closure)

        param_history.append(d['v'].cpu().detach().numpy())
        loss_history.append(loss)

        # if( epoch % print_freq == 0 ):
        #     pbar.set_postfix(**build_stats(loss, stats))
        #     pbar.update(1)
        pbar.set_postfix(**build_stats(loss, stats))
        pbar.update(1)
        if(epoch >= prof): range_pop()
        if(epoch >= prof): range_pop()

    if( prof < n_epochs ):
        torch.cuda.cudart().cudaProfilerStop()

    d.update({'param_history': param_history, 'loss_history': loss_history})
    return d

def postprocess(**d):
    p = d['plotting']
    vmin = d['v_true'].min()
    vmax = d['v_true'].max()
    
    # define the figure layout
    fig, axs = plt.subplots(3, 2, figsize=p['figsize'], 
                            gridspec_kw={'width_ratios': [1, 0.05]})
    fig.subplots_adjust(wspace=0.05, hspace=0.2)  # set the spacing between axes. 

    # plot data
    im0 = axs[0, 0].imshow(d['v_init'].cpu().T, aspect='auto', cmap=p['cmap'], vmin=vmin, vmax=vmax)
    axs[0, 0].set_title("Initial")
    fig.colorbar(im0, cax=axs[0, 1])

    im1 = axs[1, 0].imshow(d['v'].detach().cpu().T, aspect='auto', cmap=p['cmap'], vmin=vmin, vmax=vmax)
    axs[1, 0].set_title("Out")
    fig.colorbar(im1, cax=axs[1, 1])

    u = d['v_true'][:d['ny'], :d['nx']]
    im2 = axs[2, 0].imshow(u.cpu().T, aspect='auto', cmap=p['cmap'], vmin=vmin, vmax=vmax)
    axs[2, 0].set_title("True")
    fig.colorbar(im2, cax=axs[2, 1])

    plt.tight_layout()
    plt.savefig(p['output_files'][0])

    plt.close()
    make_gif(d['param_history'], 'velocities', p['cmap'])

