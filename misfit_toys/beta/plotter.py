import matplotlib.pyplot as plt


def gauss_plotter(*, data, idx, fig, axes, shape):
    plt.clf()
    plt.suptitle(
        r'$\mu={:.2f}, \sigma={:.2f}$'.format(data.mu[idx[0]], data.sig[idx[1]])
    )
    plt.subplot(*shape, 1)
    plt.plot(data.x, data.raw[idx], label='Computed PDF')
    plt.plot(
        data.x, data.pdf_ref[idx], 'ro', label='Analytic PDF', markersize=1
    )
    plt.ylim(data.raw.min(), data.raw.max())
    plt.title('PDF')

    plt.subplot(*shape, 2)
    plt.plot(data.x, data.cdf[idx], label='Computed CDF')
    plt.plot(
        data.x, data.cdf_ref[idx], 'ro', label='Analytic CDF', markersize=1
    )
    plt.title('CDF')

    plt.subplot(*shape, 3)
    plt.plot(data.p, data.Q[idx], label=r'$Q$')
    plt.plot(data.p, data.Qref[idx], 'ro', markersize=1, label=r'Analytic $Q$')
    plt.legend(framealpha=0.3)
    plt.title('Quantile')
    plt.ylim(data.x.min(), data.x.max())
    plt.tight_layout()


def w2_plotter(*, data, idx, fig, axes, shape):
    plt.clf()
    plt.suptitle(
        r'$\mu={:.2f}, \sigma={:.2f}$'.format(data.mu[idx[0]], data.sig[idx[1]])
    )
    plt.subplot(*shape, 1)
    plt.plot(data.x, data.raw[idx], label='Computed PDF')
    plt.plot(
        data.x, data.pdf_ref[idx], 'ro', label='Analytic PDF', markersize=1
    )
    plt.ylim(data.raw.min(), data.raw.max())
    plt.title('PDF')

    plt.subplot(*shape, 2)
    plt.plot(data.x, data.cdf[idx], label='Computed CDF')
    plt.plot(
        data.x, data.cdf_ref[idx], 'ro', label='Analytic CDF', markersize=1
    )
    plt.title('CDF')

    plt.subplot(*shape, 3)
    plt.plot(data.x, data.Q[idx], label=r'$Q$')
    # plt.plot(data.x, data.x, 'ro', label='Identity', markersize=3)
    plt.plot(data.x, data.Qref[idx], 'ro', markersize=5, label=r'Analytic $Q$')
    plt.legend(framealpha=0.3)
    plt.title('Quantile')
    plt.ylim(data.x.min(), data.x.max())

    plt.subplot(*shape, 4)
    plt.imshow(data.distances, cmap='gray')
    plt.colorbar()
    plt.plot([idx[1]], [idx[0]], 'r*', markersize=10)

    plt.subplot(*shape, 5)
    plt.imshow(data.distances_ref, cmap='gray')
    plt.colorbar()
    plt.plot([idx[1]], [idx[0]], 'r*', markersize=10)

    plt.subplot(*shape, 6)
    plt.imshow(data.distances_diff, cmap='gray')
    plt.colorbar()
    plt.plot([idx[1]], [idx[0]], 'r*', markersize=10)
    plt.tight_layout()


def trace_plotter(*, data, idx, fig, axes, **kw):
    int_history = data.int_history[idx[0]]
    meta = int_history.meta
    d = int_history.data
    subplot_no = 5
    for k, v in d.items():
        plt.subplot(*data.plot.sub.shape, subplot_no)
        plt.plot(meta.x, v)
        plt.title(k)
        subplot_no += 1


def trace_stack_plotter(*, data, idx, fig, axes, **kw):
    int_history = data.int_history[idx[0]]
    meta = int_history.meta
    d = int_history.data
    subplot_no = 5
    plt.subplot(*data.plot.sub.shape, subplot_no)
    style_seq = ['-', '--', '-.', ':']
    color_seq = ['b', 'r', 'g', 'c', 'm', 'y', 'k']
    num_styles = len(style_seq)
    num_colors = len(color_seq)
    curr = 0
    for k, v in d.items():
        plt.plot(
            meta.x,
            v,
            label=k,
            linestyle=style_seq[curr % num_styles],
            color=color_seq[curr % num_colors],
        )
        curr += 1
    plt.legend(framealpha=0.3)
    plt.title('Stacked Intermediate data')


def trace_group_stack_plotter(*, data, idx, fig, axes, **kw):
    d = data.int_history[idx[0]]
    subplot_no = 5
    style_seq = ['-', '--', '-.', ':']
    color_seq = ['b', 'r', 'g', 'c', 'm', 'y', 'k']
    num_styles = len(style_seq)
    num_colors = len(color_seq)
    for k, v in d.items():
        curr = 0
        plt.subplot(*data.plot.sub.shape, subplot_no)
        for kk, vv in v.items():
            if kk == 'x':
                continue
            plt.plot(
                v.x,
                vv,
                label=kk,
                linestyle=style_seq[curr % num_styles],
                color=color_seq[curr % num_colors],
            )
            curr += 1
        plt.title(k)
        plt.legend(framealpha=0.3)
        subplot_no += 1
