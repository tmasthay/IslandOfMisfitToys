import matplotlib.pyplot as plt
from masthay_helpers.typlotlib import setup_gg_plot, rand_color, set_color_plot_global
import torch
import os
import deepwave

from .elastic_class import *
from .seismic_data import marmousi_real

setup_gg_plot(clr_out='black', clr_in='black')
config_plot = set_color_plot_global(
    use_legend=False, 
    use_colorbar=True, 
    use_grid=False
)

def get_cmd_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sub', type=int, default=1)
    parser.add_argument('--dynamic', action='store_true')
    parser.add_argument('--cmap', type=str, default='seismic')
    parser.add_argument('--mode', type=str, default='time_offset')
    parser.add_argument('--recompute', action='store_true')

    args = parser.parse_args()
    return args

def get_data(data_call, args, **kw):
    print('BUILDING MARMOUSI OBJECT...', end='')
    data = data_call(**kw)
    print('BUILT')
    u = None
    if( not os.path.exists('u.pt') or args.recompute ):
        print(
            'COMPUTING wavefield...CTRL+C NOW TO ABORT...', 
            end='', 
            file=sys.stderr
        )
        u = data.forward()
        print('COMPUTED', file=sys.stderr)
        print('SAVING...', end='')
        torch.save(u, 'u.pt')
        print('SAVED')
        
    if( u == None ):
        print('LOADING...', end='')
        u = torch.load('u.pt')
        print('LOADED pytorch binary')

    print('TRANSFERRING...', end='')
    u = u.cpu()
    print('TRANSFERRED pytorch binary to CPU')
    return u, data

def get_plot_config(u, args):
    kw = {
        'cmap': args.cmap,
        'aspect': 'auto'
    }

    config_plot = set_color_plot_global(
        use_legend=False, 
        use_colorbar=True, 
        use_grid=False,
        colorbar_kw={
            'label': 'Displacement',
            'labelcolor': 'white'
        }
    )
    print('PLOTTING...', end='')
    open_plot = open_ide(
        ('code', 'code'),    # Visual Studio Code
        ('atom', 'atom'),    # Atom
        ide_precedence=True,
        no_ide=['display', 'feh', 'eog', 'xdg-open'],
        default='/usr/bin/open'
    )
    setup_gg_plot(clr_out='black', clr_in='black')
    return kw, open_plot, config_plot

def plot_2d(
    *,
    vals, 
    title, 
    name, 
    config_plot, 
    open_plot, 
    exclude_keys=[],
    appendage={},
    commands=[
        lambda: plt.xlabel('Horizontal location (m)'),
        lambda: plt.ylabel('Depth (m)')
    ],
    **kw
):
    name = name.replace('.jpg', '')
    plt.imshow(vals, **kw)
    config_plot(
        title,
        exclude_keys=exclude_keys,
        appendage=appendage,
        commands=commands
    )
    plt.savefig('%s.jpg'%name)
    plt.clf()
    open_plot('%s.jpg'%name)

def plot_1d_sequence(
    *,
    vals, 
    title, 
    name, 
    config_plot, 
    open_plot, 
    dt,
    dx,
    step=1,
    exclude_keys=['use_colorbar', 'colorbar_kw', 'cmap'],
    appendage={},
    commands=[
        lambda: plt.xlabel('Time (s)'),
        lambda: plt.ylabel('Displacement Amplitude (m)')
    ],
    **kw,
):
    kwargs = {**{k: kw[k] for k in kw.keys()}, **appendage}
    commands.append(lambda: plt.ylim(vals.min(), vals.max()))
    name = name.replace('.jpg', '')
    t = [i * dt for i in range(0, len(vals[0]))]
    nx = vals.shape[0]
    for i in range(0, nx, step):
        print('Plotting %d of %d'%(i // step, nx // step))
        plt.plot(t, vals[i,:], **kwargs)
        config_plot(
            '%s x=%d'%(title, i*dx),
            exclude_keys=exclude_keys,
            appendage=appendage,
            commands=commands
        ) 
        plt.savefig('%s_%d.jpg'%(name, i))
        plt.clf()
    os.system(f'convert -delay 10 $(ls -tr {name}_*.jpg) {name}.gif')
    os.system(f'rm {name}_*.jpg')
    open_plot(f'{name}.gif')

def plot_results(
    *, 
    displacement, 
    data, 
    config_plot,
    open_plot,
    dt,
    dx,
    step=1,
    **kw
):
    def helper(*, func, **kw_local):
        func(open_plot=open_plot, config_plot=config_plot, **kw_local)

    helper(
        func=plot_2d,
        vals=torch.transpose(displacement[0], 0, 1).cpu(),
        title='Time offset',
        name='u_time_offset',
        commands=[
            lambda: plt.xlabel('Horizontal location (m)'),
            lambda: plt.ylabel('Time (s)')
        ],
        **{
            **kw,
            'extent': [0, data.nx*data.dx, data.nt*data.dt, 0.0],
            'vmin': displacement.min(),
            'vmax': displacement.max()
        }
    )
    helper(
        func=plot_2d,
        vals=data.vp.cpu(),
        title=r'$V_p$',
        name='vp',
        **{
            **kw,
            'extent': [0, data.nx*data.dx, data.ny*data.dy, 0.0],
        }
    )
    plot_1d_sequence(
        vals=displacement[0],
        title='Trace',
        name='u_trace',
        dt=dt,
        dx=dx,
        open_plot=open_plot,
        config_plot=config_plot,
        step=step
    )

if( __name__ == '__main__' ):
    args = get_cmd_line_args()
    u, data = get_data(marmousi_real, args)
    plot_kw, open_plot, config_plot = get_plot_config(u, args)
    plot_results(
        displacement=u,
        data=data,
        config_plot=config_plot,
        open_plot=open_plot,
        dt=data.dt,
        dx=data.dx,
        step=100,
        **plot_kw
    )