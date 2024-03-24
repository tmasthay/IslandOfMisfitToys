import os
import sys

import matplotlib.pyplot as plt
import torch


def verify_and_plot(self, *, plotter, name, computed, ref, **kw):
    make_plots = plotter(self, name=name, computed=computed, ref=ref, **kw)
    # mse = torch.nn.functional.mse_loss(computed, ref)
    try:
        torch.testing.assert_close(
            computed, ref, rtol=self.c.rtol, atol=self.c.atol
        )
        if self.c.plot.mode.lower() in ['always', 'success']:
            # print(f'SUCCESS: {mse:.2e=}', flush=True)
            print('SUCCESS', flush=True)
            make_plots('SUCCESS')
    except AssertionError as e:
        if self.c.plot.mode.lower() in ['always', 'fail']:
            mse = torch.nn.functional.mse_loss(computed, ref)
            print(f'FAILURE: {mse:.2e=}', flush=True)
            make_plots('FAILURE')
        raise e
    except Exception as e:
        raise e


def should_plot(self, name, *, status, mu, sigma):
    path = os.path.abspath(os.path.dirname(__file__))
    already_plotted = [
        e for e in os.listdir(path) if e.startswith(name) and e.endswith('.jpg')
    ]
    # all_params = f'{path}/{name}_'
    # print(all_params, flush=True)
    # sys.exit(-1)
    # for k, v in kw.items():
    #     print(all_params, flush=True)
    #     all_params += f'{k}={v}_'
    # all_params += f'{status}.jpg'
    # path = all_params
    # print(f'{path=}', flush=True)
    path = f'{path}/{name}_{mu:.2f}_{sigma:.2f}_{status}.jpg'
    if len(already_plotted) >= self.c.plot.max or os.path.exists(path):
        return ''
    return path


def plot_quantile(self, *, name, x, computed, ref, mu, sigma):
    def helper(status):
        path = should_plot(self, name, status=status, mu=mu, sigma=sigma)
        if not path:
            return

        plt.clf()
        plt.plot(self.p, computed, label='computed')
        plt.plot(self.p, ref, label='analytic', linestyle='--')
        plt.legend()
        plt.title(rf'$\mu = {mu:.2f}, \sigma = {sigma:.2f}, status = {status}$')
        plt.ylim(x[0], x[-1])
        plt.savefig(path)

    return helper


def plot_cdf(self, *, name, x, computed, ref, mu, sigma):
    def helper(status):
        path = should_plot(self, name, mu=mu, sigma=sigma, status=status)
        if not path:
            return
        plt.clf()
        plt.plot(x, computed, label='computed')
        plt.plot(x, ref, label='analytic', linestyle='--')
        plt.legend()
        plt.title(
            r'$\mu = %.2f, \sigma = %.2f, status = %s$' % (mu, sigma, status)
        )
        plt.ylim(0, 1)
        plt.savefig(path)

    return helper


def plot_pdf(self, *, name, x, computed, ref, mu, sigma):
    def helper(status):
        path = should_plot(self, name, mu=mu, sigma=sigma, status=status)
        if not path:
            return
        plt.clf()
        plt.plot(x, computed, label='computed')
        plt.plot(x, ref, label='analytic', linestyle='--')
        plt.legend()
        plt.title(
            r'$\mu = %.2f, \sigma = %.2f, status = %s$' % (mu, sigma, status)
        )
        plt.ylim(0, 1)
        plt.savefig(path)

    return helper
