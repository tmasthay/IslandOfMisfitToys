import pytest
import torch


@pytest.fixture(scope='session')
def sine_ref_data(cfg, adjust):
    def helper(freq):
        c = cfg.unit.beta.prob.unbatch_splines
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
