from dataclasses import dataclass

import numpy as np
import pytest
import torch

# from hypothesis import given, settings
# from hypothesis import strategies as st
from mh.core import DotDict, draise
from plot_loss import plot_loss, verify_and_plot
from pytest import mark


@dataclass
class ConvLocal:
    max_examples: int = 1


# @mark.dummy
# def test_dummy(cfg):
#     import yaml
#     from mh.core import colorize_yaml
#     from rich.console import Console

#     console = Console(file=open('output.txt', 'w'))
#     s = yaml.dump(cfg)
#     s = s.replace('!!python/object:mh.core.DotDict', '')
#     colors = [
#         'bright_cyan',
#         'bright_magenta',
#         'bright_blue',
#         'bright_green',
#         'bright_yellow',
#         'orange3',
#         'medium_purple3',
#         'light_coral',
#         'gold3',
#     ]
#     depth_color_map = {i: color for i, color in enumerate(colors)}
#     s = colorize_yaml(s.split('\n'), depth_color_map, 'red')

#     console.print(s)
#     return True


@mark.mysub
def test_loss(cfg, subtests):
    c = cfg.unit.beta.conv

    def run_case(i, *, d):
        # draise(data)
        data = d.data.to(d.device)
        init_guess = d.init_guess.to(d.device)
        init_guess.requires_grad = True
        optimizer = d.optimizer([init_guess])
        loss_fn_fixed = d.loss_fn(data)
        with subtests.test(i=i, msg=d.name):
            loss_history = torch.empty(c.nepochs, device=d.device)
            soln_history = torch.empty(
                c.nepochs, *init_guess.shape, device=d.device
            )
            grad_history = torch.empty(
                c.nepochs, *init_guess.shape, device=d.device
            )
            mse_history = torch.empty(c.nepochs, device=d.device)
            int_history = [None for _ in range(c.nepochs)]
            print('\n\n')
            for epoch in range(c.nepochs):
                num_calls = 0

                def closure():
                    nonlocal num_calls
                    num_calls += 1
                    optimizer.zero_grad()
                    res = loss_fn_fixed(init_guess)
                    if type(res) == tuple:
                        loss, intermediate = res
                        if intermediate is not None:
                            if type(intermediate) is not DotDict:
                                raise TypeError(
                                    'Intermediate values must be DotDicts'
                                    f'got {type(intermediate)} instead'
                                )
                    else:
                        loss = res
                        intermediate = None
                    loss.backward()
                    if num_calls == 1:
                        grad_history[epoch] = (
                            init_guess.grad.detach().clone().cpu()
                        )
                        soln_history[epoch] = init_guess.detach().clone().cpu()
                        loss_history[epoch] = loss.detach().clone().cpu()
                        mse_history[epoch] = (
                            torch.mean((init_guess - data) ** 2)
                            .detach()
                            .clone()
                            .cpu()
                        )
                        int_history[epoch] = intermediate
                        freq = torch.inf
                        end_char = '\n' if epoch % freq == 0 else '\r'
                        print(
                            f'epoch = {epoch:06d}, '
                            f'loss = {loss_history[epoch].item():.2e}, '
                            f'grad_norm={torch.norm(grad_history[epoch]):.2e}, '
                            f'mse = {mse_history[epoch].item():.2e}',
                            flush=True,
                            end=end_char,
                        )
                    return loss

                optimizer.step(closure)

            d.data = d.data.to('cpu')
            d.plot.iter.strides = [loss_history.shape[0] // c.num_plots, 1]
            all_none = all([e is None for e in int_history])
            has_int_plotter = d.plot.get('int_plotter', None) is not None
            if all_none and has_int_plotter:
                raise ValueError(
                    'int_plotter supplied but all None intermediate valuesfound'
                )
            elif not all_none and not has_int_plotter:
                raise ValueError(
                    'int_plotter not supplied but non-None intermediate values'
                    'found'
                )
            d_pass = DotDict(
                dict(
                    loss_history=loss_history.detach().cpu(),
                    soln_history=soln_history.detach().cpu(),
                    grad_history=grad_history.detach().cpu(),
                    mse_history=mse_history.detach().cpu(),
                    int_history=int_history,
                    t=c.x.detach().cpu(),
                    out_path=cfg.hydra_out,
                    **d,
                )
            )
            print('\n\n')
            verify_and_plot(plotter=plot_loss, d=d_pass)

    cases = c.subtests
    for k, case in cases.items():
        i = int(k)
        run_case(i, d=case)
