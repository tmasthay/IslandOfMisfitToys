from dataclasses import dataclass

import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from plot_prob import *
from plot_w2 import plot_w2, verify_and_plot
from pytest import mark

from misfit_toys.beta.prob import cdf, pdf
from misfit_toys.beta.w2 import w2


@dataclass
class W2Local:
    max_examples: int = 10


@mark.sub
def test_loss(cfg, subtests):
    c = cfg.unit.beta.conv

    def run_case(i, *, loss_fn, optimizer, data):
        with subtests.test(
            i=i, msg=f'{loss_fn.__name__}, {optimizer.__name__}'
        ):
            for epoch in range(c.epochs):
                optimizer.zero_grad()
                loss = loss_fn()
                loss.backward()
                optimizer.step()
