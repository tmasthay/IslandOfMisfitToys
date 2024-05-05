import os

import matplotlib.pyplot as plt
import torch


def verify_and_plot(self, *, plotter, name, computed, ref, **kw):
    print('hello', flush=True)
    make_plots = plotter(self, name=name, computed=computed, ref=ref, **kw)
    # mse = torch.nn.functional.mse_loss(computed, ref)
    try:
        torch.testing.assert_close(
            computed, ref, rtol=self.c.rtol, atol=self.c.atol
        )
        # exc = kw.get('exc', set([]))
        # s = ', '.join([f'{k}={v}' for k, v in kw.items() if k not in exc])
        if self.c.plot.mode.lower() in ['always', 'success']:
            print('SUCCESS', flush=True)
            make_plots('SUCCESS')
    except AssertionError as e:
        try:
            if self.c.plot.mode.lower() in ['always', 'fail']:
                print('FAILURE', flush=True)
                make_plots('FAILURE')
        except Exception as other_e:
            print(
                f'Unexpected failure in plotting failure: {other_e}', flush=True
            )
            raise other_e
        raise e
    except Exception as e:
        print(f'Unexpected exception in SUCCESS branch: {e}', flush=True)
        raise e


def should_plot(self, name, *, status, **kw):
    path = os.path.abspath(os.path.dirname(__file__))
    already_plotted = [
        e for e in os.listdir(path) if e.startswith(name) and e.endswith('.jpg')
    ]
    path = f'{path}/{name}_'
    kw['exclusions'] = kw.get('exclusions', [])
    keys = set(kw.keys()) - set(kw['exclusions'] + ['exclusions'])
    for k in keys:
        v = kw[k]
        s = str(v)
        if s.startswith('<') and s.endswith('>'):
            s = s.split(' ')[0].upper()
        path += f'{k}={s}_'
    path += f'{status}.jpg'
    if len(already_plotted) >= self.c.plot.max or os.path.exists(path):
        return ''
    return path


def plot_w2(
    self,
    *,
    name,
    x,
    computed,
    ref,
    mu1,
    sigma1,
    mu2,
    sigma2,
    Qref,
    Qcomputed,
    Tref,
    Tcomputed,
    cdfRef,
    cdfComputed,
    exclusions,
):
    def helper(status):
        path = should_plot(
            self,
            name,
            mu1=mu1,
            sigma1=sigma1,
            mu2=mu2,
            sigma2=sigma2,
            status=status,
            Qref=Qref,
            Qcomputed=Qcomputed,
            Tref=Tref,
            Tcomputed=Tcomputed,
            cdfComputed=cdfComputed,
            cdfRef=cdfRef,
            exclusions=exclusions,
        )
        # print(path, flush=True)
        if not path:
            return

        err = torch.abs(computed - ref).mean()
        plt.clf()
        plt.subplot(3, 1, 1)
        plt.plot(self.p, Qcomputed, label='computed')
        plt.plot(self.p, Qref, label='analytic', linestyle='--')
        plt.legend()
        plt.suptitle(
            f'Comp={computed:.2e}, Ref={ref:.2e}, '
            rf'$\epsilon = {err:.2e}$'
            '\n'
            rf'$\mu_1 = {mu1:.2f}, \sigma_1 = {sigma1:.2f}'
            rf', \mu_2 = {mu2:.2f}, \sigma_2 = {sigma2:.2f},'
            rf' status = {status}$'
        )
        plt.title('Quantile')

        plt.subplot(3, 1, 2)
        plt.plot(x, Tcomputed, label='computed')
        plt.plot(x, Tref, label='analytic', linestyle='--')
        # plt.title('Integrand: ' + r'$(t - G^{-1}(F(t))^2 f(t)$')
        plt.title('Integrand')
        plt.legend()

        plt.subplot(3, 1, 3)
        plt.plot(x, cdfComputed, label='computed')
        plt.plot(x, cdfRef, label='analytic', linestyle='--')
        plt.title('CDF')
        plt.legend()
        plt.tight_layout()
        plt.savefig(path)

        print(f'{status} written to\n    {os.path.abspath(path)}', flush=True)

    return helper
