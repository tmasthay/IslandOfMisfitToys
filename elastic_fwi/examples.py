from elastic_class import *
import matplotlib.pyplot as plt
from helpers.typlotlib import setup_gg_plot, rand_color, set_color_plot_global
import torch
import os
import deepwave
from seismic_data import marmousi_real

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
        'aspect': 'auto',
        'extent': [0, data.ny*data.dy, data.nx*data.dx, 0],
        'vmin': u.min(),
        'vmax': u.max()
    }
    if( args.dynamic ):
        kw.pop('vmin'), kw.pop('vmax')

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
    xlabel='',
    ylabel='',
    **kw
):
    name = name.replace('.jpg', '')
    plt.imshow(vals, **kw)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    config_plot(title)
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
    xlabel='',
    ylabel='',
    step=1,
    **kw,
):
    name = name.replace('.jpg', '')
    t = [i * dt for i in range(0, len(vals[0]))]
    nx = vals.shape[0]
    for i in range(0, nx, step):
        print('Plotting %d of %d'%(i // step, nx // step))
        plt.plot(t, vals[i,:], **kw)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        config_plot('%s x=%d'%(title, i*dx))
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
        vals=displacement[0],
        title='Time offset',
        name='u_time_offset',
        xlabel='Horizontal location (km)',
        ylabel='Depth (km)',
        **kw
    )
    helper(
        func=plot_2d,
        vals=data.vp.cpu(),
        title=r'$V_p$',
        name='vp',
        xlabel='Horizontal location (km)',
        ylabel='Depth (km)',
        **kw
    )
    plot_1d_sequence(
        vals=displacement[0],
        title='Trace',
        name='u_trace',
        xlabel='Time (s)',
        ylabel='Amplitude',
        dt=dt,
        dx=dx,
        open_plot=open_plot,
        config_plot=(lambda x: plt.title(x)),
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
