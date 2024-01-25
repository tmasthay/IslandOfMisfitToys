import os

import matplotlib.pyplot as plt
import pytest
import torch
from gen_data import gen_data
from masthay_helpers.global_helpers import DotDict, hydra_kw
from masthay_helpers.typlotlib import plot_tensor2d_fast, plot_tensor2d_subplot
from scipy.special import erfinv

from misfit_toys.fwi.loss.w2 import (
    cts_quantile,
    unbatch_spline_eval,
    w2,
    w2_const,
)


# Fixture to generate data
@pytest.fixture(scope='session')
def tdata():
    return gen_data(config_path=os.getcwd(), config_name='config')


def w2_return_val(d):
    quantiles = cts_quantile(d.gauss, d.x, d.p, dx=d.x[1] - d.x[0])
    w2_dist_squared = w2_const(d.gauss, d.x, quantiles=quantiles)
    return w2_dist_squared


def test_quantile(tdata):
    quantiles = cts_quantile(
        tdata.gauss, tdata.x, tdata.p, dx=tdata.x[1] - tdata.x[0]
    )
    res = unbatch_spline_eval(quantiles, tdata.p)
    print(res)


@hydra_kw(use_cfg=True)
def main(cfg: DotDict):
    cfg = cfg.gauss
    d = gen_data(config_path=os.getcwd(), config_name='config')

    w2_data = w2_return_val(d.gauss)

    def config(x):
        plt.title(x)
        plt.colorbar()

    plot_tensor2d_fast(
        tensor=d.gauss.gauss,
        labels=['x', 'shift', 'scale'],
        name='gauss',
        print_freq=1,
        verbose=True,
        config=config,
        extent=[-cfg.shift, cfg.shift, cfg.support[0], cfg.support[1]],
    )

    d.gauss.expected_output = d.gauss.expected_output.permute(1, 0, 2)
    plot_tensor2d_fast(
        tensor=d.gauss.expected_output,
        labels=['shift', 'scale'],
        name='expected_output',
        print_freq=1,
        verbose=True,
        config=config,
        extent=[cfg.scales[0], cfg.scales[1], -cfg.shift, cfg.shift],
    )

    plot_tensor2d_fast(
        tensor=w2_data,
        labels=['shift', 'scale'],
        name='w2_data',
        print_freq=1,
        verbose=True,
        config=config,
        extent=[cfg.scales[0], cfg.scales[1], -cfg.shift, cfg.shift],
    )


if __name__ == "__main__":
    main(config_path=os.getcwd(), config_name='config')
