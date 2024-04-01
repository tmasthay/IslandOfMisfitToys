import pytest
import torch

from misfit_toys.beta.prob import *


@pytest.fixture(scope='session')
def sine_ref_data(cfg, adjust):
    def helper(freq):
        c = cfg.unit.beta.unbatch_splines
        freq = adjust(freq, *c.freq)

        x = torch.linspace(c.x.left, c.x.right, c.x.num_ref)
        y = torch.sin(freq * x)

        x_test = torch.linspace(c.x.left, c.x.right, c.x.num_ref * c.x.upsample)
        x_test = x_test[c.pad : -c.pad]
        y_true = torch.sin(freq * x_test)
        y_deriv_true = freq * torch.cos(freq * x_test)

        atol = c.get('atol', cfg.atol)
        rtol = c.get('rtol', cfg.rtol)
        return x, y, x_test, y_true, y_deriv_true, atol, rtol

    return helper


@pytest.fixture(scope='session')
def gauss_pdf_computed():
    def helper(x_args, mu, sigma):
        # hack way around this for now
        #     -> you have an inconsistency in your yamls
        #     and this fixture is used in multiple tests

        if isinstance(x_args, torch.Tensor):
            x = x_args
        else:
            x = torch.linspace(*x_args)
        z = torch.exp(-((x - mu) ** 2) / (2 * sigma**2))

        def renorm(y1, x1):
            return y1 / torch.trapz(y1, x1, dim=-1).unsqueeze(-1)

        z = pdf(z, x, renorm=renorm)
        return z, x

    return helper


@pytest.fixture(scope='session')
def gauss_pdf_ref():
    def helper(x, mu, sigma):
        z = (
            1
            / (sigma * np.sqrt(2 * np.pi))
            * torch.exp(-((x - mu) ** 2) / (2 * sigma**2))
        )
        return z

    return helper


@pytest.fixture(scope='session')
def gauss_cdf_ref():
    def helper(x, mu, sigma):
        z = 0.5 * (1 + torch.erf((x - mu) / (sigma * np.sqrt(2))))
        return z

    return helper


@pytest.fixture(scope='session')
def gauss_quantile_ref():
    def helper(p, mu, sigma):
        z = mu + sigma * np.sqrt(2) * torch.erfinv(2 * p - 1)
        return z

    return helper
