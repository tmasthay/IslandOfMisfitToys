import os

import matplotlib.pyplot as plt
import torch
from mh.core import convert_dictconfig, hydra_kw
from mh.typlotlib import get_frames_bool, save_frames

from misfit_toys.fwi.loss.w2 import cum_trap, true_quantile, unbatch_spline_eval
from misfit_toys.utils import bool_slice, clean_idx


@hydra_kw(use_cfg=True)
def plot_eval(c, d, exp_output, exp_grad, *, pytest_cfg):
    # c = convert_config_correct(c)
    pc = pytest_cfg
    c = convert_dictconfig(c)
    c.order = (
        list(range(1, 1 + c.sub.shape[0] * c.sub.shape[1]))
        if c.order is None
        else c.order
    )

    # Q = unbatch_spline_eval(d.q, pc.p)
    # Qd = unbatch_spline_eval(d.qd, pc.p)
    Q, Qd = d.q(pc.p), d.qd(pc.p)

    shift_expand = pc.shift.unsqueeze(-1).expand(
        pc.shift.shape[0], pc.scale.shape[0]
    )
    scale_expand = pc.scale.unsqueeze(0).expand(
        pc.shift.shape[0], pc.scale.shape[0]
    )

    Qdirect = true_quantile(
        d.uniform_pdfs,
        pc.t,
        pc.p,
        ltol=pc.ltol,
        rtol=pc.rtol,
        atol=pc.atol,
        err_top=pc.err_top,
    )
    # raise ValueError(d.uniform_pdfs.shape)

    fig, axes = plt.subplots(*c.sub.shape, **c.sub.kw)
    plt.subplots_adjust(**c.sub.adjust)
    plt.suptitle(c.suptitle)

    def plotter(*, data, idx, fig, axes):
        def set_plot(i):
            plt.subplot(*c.sub.shape, c.order[i - 1])

        curr_shift = shift_expand[idx[0], idx[1]]
        curr_scale = scale_expand[idx[0], idx[1]]
        subtitle = f'$U([{curr_shift:.1f}, {curr_shift + curr_scale:.1f}])$'
        plt.clf()
        plt.suptitle(f'{c.suptitle}\n{subtitle}')
        set_plot(1)
        plt.imshow(
            d.misfit.res,
            **c.imshow.kw,
            extent=[
                pc.scale.min(),
                pc.scale.max(),
                pc.shift.min(),
                pc.shift.max(),
            ],
        )
        # y = [curr_shift, curr_shift]
        # x = [pc.scale.min(), pc.scale.max()]
        y = [pc.shift.min(), pc.shift.max()]
        x = [curr_scale, curr_scale]
        plt.plot(x, y, color='orange', linestyle='-.')
        plt.plot([curr_scale], [curr_shift], 'x', color='red', markersize=10)
        plt.xlabel('Scale')
        plt.ylabel('Shift')
        plt.title('Misfit')
        plt.colorbar()

        set_plot(2)
        plt.plot(pc.shift, d.misfit.res[:, idx[1]], **c.opts[0])
        plt.plot(pc.shift, exp_output[:, idx[1]], **c.opts[1])
        plt.plot(
            [curr_shift],
            [d.misfit.res[idx[0], idx[1]]],
            'o',
            color='green',
            markersize=10,
        )
        plt.legend(**c.legend[0])
        plt.xlabel('Shift')
        plt.ylabel('Misfit')
        plt.title(f'Misfit vs Shift: {d.misfit.res[:, idx[1]].min()}')
        plt.ylim(
            min(d.misfit.res.min(), exp_output.min()),
            max(d.misfit.res.max(), exp_output.max()),
        )

        set_plot(3)
        plt.plot(pc.p, Q[idx], **c.opts[0])
        plt.plot(
            pc.p,
            pc.p * curr_scale + curr_shift,
            **c.opts[1],
        )
        # plt.plot(pc.p, Qdirect[idx], **c.opts[1])
        plt.ylim(min(Q.min(), Qdirect.min()), max(Q.max(), Qdirect.max()))
        plt.xlabel('p')
        plt.ylabel('Q')
        plt.title('Quantile')
        plt.legend(**c.legend[0])

        set_plot(4)
        plt.plot(pc.t, d.uniform_pdfs[idx], **c.opts[0])
        plt.plot(pc.t, d.ref_pdf, **c.opts[1])
        plt.ylim(d.uniform_pdfs.min(), d.uniform_pdfs.max())
        plt.xlabel('t')
        plt.ylabel('PDF')
        plt.title('PDF')

        set_plot(5)
        # num_deriv = torch.diff(Qd[idx]) / torch.diff(pc.p)
        plt.plot(pc.p, Qd[idx], **c.opts[0])
        plt.plot(
            pc.p,
            curr_scale * torch.ones_like(pc.p),
            **c.opts[1],
        )
        # plt.plot(
        #     pc.p[:-1],
        #     num_deriv,
        #     **c.opts[2],
        # )
        plt.xlabel('p')
        plt.ylabel('Qd')
        plt.title('Quantile Derivative')
        plt.ylim(0.0, 5.0)
        plt.legend(**c.legend[0])

        set_plot(6)
        plt.plot(pc.t, d.misfit.transport[idx], **c.opts[0])
        # plt.plot(
        #     pc.t,
        #     (pc.t - d.misfit.transport[idx]) ** 2
        #     * pc.t
        #     * (pc.t >= 0)
        #     * (pc.t <= 1),
        #     **c.opts[3],
        # )

        transport_opts = {k: v for k, v in c.opts[1].items()}
        transport_opts['label'] = 'Identity'
        plt.plot(pc.t, pc.t, **transport_opts)
        plt.ylim(pc.t.min(), pc.t.max())
        plt.xlabel('t')
        plt.ylabel('Transport')
        plt.title('Transport')
        plt.legend(**c.legend[0])
        # plt.xlim(-0.1, 0.1)

        set_plot(7)
        plt.plot(pc.t, cum_trap(d.uniform_pdfs[idx], pc.t), **c.opts[0])
        plt.plot(
            pc.t,
            ((pc.t - curr_shift) / curr_scale)
            * (pc.t >= curr_shift)
            * (pc.t <= curr_shift + curr_scale)
            + (pc.t > curr_shift + curr_scale),
            **c.opts[1],
        )
        plt.title('CDF')
        plt.legend(**c.legend[0])

        set_plot(8)
        # u = -d.grad[idx]
        plt.plot(pc.t, -d.grad[idx], **c.opts[0])
        plt.plot(pc.t, exp_grad[idx], **c.opts[1])
        plt.ylim(-d.grad.max(), -d.grad.min())
        plt.legend(**c.legend[0])

        plt.tight_layout()

        set_plot(9)
        plt.imshow(
            torch.abs(exp_output - d.misfit.res),
            **c.imshow.kw,
            extent=[
                pc.scale.min(),
                pc.scale.max(),
                pc.shift.min(),
                pc.shift.max(),
            ],
        )
        plt.colorbar()
        plt.title('Absolute Error')
        plt.xlabel('Scale')
        plt.ylabel('Shift')
        return {}

    iter = bool_slice(*Q.shape, **c.iter[0])
    frames = get_frames_bool(
        data=None, iter=iter, fig=fig, axes=axes, plotter=plotter
    )
    save_frames(frames, **c.save)
