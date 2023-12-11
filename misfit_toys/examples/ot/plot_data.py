from masthay_helpers.typlotlib import make_gifs
import os


def plot_data():
    in_dir = os.path.abspath(os.path.join(__file__, '..', 'out'))
    out_dir = os.path.join(in_dir, 'figs')
    common = {
        'verbose': True,
        'duration': 250,
        'print_freq': 10,
        'path': out_dir,
    }
    verbose = True
    duration = 100
    opts = {
        'loss_record': {
            'labels': ['Epoch', 'Loss'],
        },
        'vp_record': {
            'labels': ['Extent', 'Depth', 'Epoch'],
            'permute': (2, 1, 0),
        },
        'out_record': {
            'labels': ['Extent', 'Time', 'Shot No', 'Epoch'],
            'permute': (3, 2, 1, 0),
        },
    }
    opts['out_filt_record'] = opts['out_record']
    for k in opts.keys():
        opts[k].update(common)

    make_gifs(
        in_dir=in_dir,
        out_dir=out_dir,
        targets=list(opts.keys()),
        opts=opts,
        cmap='seismic',
    )


if __name__ == '__main__':
    plot_data()
