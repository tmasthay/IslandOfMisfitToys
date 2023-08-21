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
        use_grid=False
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

if( __name__ == '__main__' ):
    args = get_cmd_line_args()
    u, data = get_data(marmousi_real, args)
    plot_kw, open_plot, config_plot = get_plot_config(u, args)

    if( args.mode == 'time_offset' ):
        plt.imshow(u[0], cmap=args.cmap, aspect='auto')
        config_plot('Time offset')
        plt.savefig("u_time_offset.jpg")
        plt.clf()
        open_plot('u_time_offset.jpg')
    elif( args.mode == 'traces' or args.mode == 'trace' ):
        plt.imshow(u[0], cmap=args.cmap, aspect='auto')
        config_plot('Time offset')
        plt.savefig("u_trace.jpg")
        plt.clf()
        open_plot('u_trace.jpg')
    else:
        print(f'Mode "{args.mode}" not supported')
    print('DONE')
