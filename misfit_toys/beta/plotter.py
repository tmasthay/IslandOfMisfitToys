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
    return {'shape': shape}
