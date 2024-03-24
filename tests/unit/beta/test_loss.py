from dataclasses import dataclass

import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from mh.core import draise
from plot_prob import *
from plot_w2 import plot_w2, verify_and_plot
from pytest import mark

from misfit_toys.beta.prob import cdf, pdf
from misfit_toys.beta.w2 import w2


@dataclass
class ConvLocal:
    max_examples: int = 10


@mark.mysub
def test_loss(cfg, subtests):
    c = cfg.unit.beta.conv

    def run_case(i, *, loss_fn, optimizer, data, init_guess, device):
        draise(data)
        data = data.to(device)
        init_guess = init_guess.to(device)
        init_guess.requires_grad = True
        optimizer = optimizer([init_guess])
        with subtests.test(
            i=i, msg=f'{loss_fn.__name__}, {optimizer.__name__}'
        ):
            loss_history = torch.empty(c.epochs, device=device)
            soln_history = torch.empty(
                c.epochs, *init_guess.shape, device=device
            )
            grad_history = torch.empty(
                c.epochs, *init_guess.shape, device=device
            )
            for epoch in range(c.epochs):
                num_calls = 0

                def closure():
                    nonlocal num_calls
                    num_calls += 1
                    optimizer.zero_grad()
                    loss = loss_fn(data, init_guess)
                    loss.backward()
                    if num_calls == 1:
                        grad_history[epoch] = (
                            init_guess.grad.detach().clone().cpu()
                        )
                        soln_history[epoch] = init_guess.detach().clone().cpu()
                        loss_history[epoch] = loss.detach().clone().cpu()
                    optimizer.step()
                    return loss

                optimizer.step(closure)

                err = loss_history[-1]
                assert err <= c.atol, f'err = {err} > {c.atol}'

    cases = c.subtests
    for i, case in enumerate(cases):
        run_case(i, **case)
