import os

import deepwave as dw
import hydra
import matplotlib.pyplot as plt
import torch
import yaml
from mh.core import DotDict, hydra_out, set_print_options, torch_stats
from mh.typlotlib import get_frames_bool, save_frames
from omegaconf import OmegaConf

from misfit_toys.utils import exec_imports, runtime_reduce

set_print_options(callback=torch_stats('all'))


def check_shape(data, shape, field, custom_str=''):
    if shape is None:
        return
    if data.shape != shape:
        raise ValueError(
            f"Expected shape {shape} for {field}, got"
            f" {data.shape}\n{custom_str}"
        )


def pretty_dict(d, depth=0, indent_str='  ', s=''):
    for k, v in d.items():
        if isinstance(v, dict) or isinstance(v, DotDict):
            s += f'{indent_str*depth}{k}:\n' + pretty_dict(
                v, depth + 1, indent_str
            )
        else:
            s += f'{indent_str*depth}{k}: {v}\n'
    return s


@hydra.main(config_path="cfg", config_name="cfg", version_base=None)
def main(cfg):
    c = DotDict(OmegaConf.to_container(cfg, resolve=True))
    c = exec_imports(c)

    def runtime_reduce_simple(key, expected_shape):
        resolve_rules = c[key].get('resolve', c.resolve)
        self_key = f'slf_{key.split(".")[-1]}'
        field_name = key.split('.')[-1]
        before_reduction = f'\n\nBefore reduction:\n{pretty_dict(c[key])}'
        c[key] = runtime_reduce(c[key], **resolve_rules, self_key=self_key)
        c[key] = c[key].to(c.device)
        check_shape(c[key], expected_shape, field_name, before_reduction)

    def full_runtime_reduce(lcl_cfg, **kw):
        return runtime_reduce(lcl_cfg, **{**lcl_cfg.resolve, **kw})

    runtime_reduce_simple('data.vp', (c.ny, c.nx))
    runtime_reduce_simple('data.rec_loc_y', (c.n_shots, c.rec_per_shot, 2))
    runtime_reduce_simple('data.src_loc_y', (c.n_shots, c.src_per_shot, 2))
    runtime_reduce_simple('data.src_amp_y', (c.n_shots, c.src_per_shot, c.nt))
    # runtime_reduce_simple('data.src_amp_y_init')
    runtime_reduce_simple('data.gbl_rec_loc', None)

    c = full_runtime_reduce(
        c, self_key='slf_gbl_obs_data', call_key="__call_gbl_obs__"
    )
    c = full_runtime_reduce(c, self_key='slf_obs_data', call_key="__call_obs__")
    check_shape(c.data.obs_data, (c.n_shots, c.rec_per_shot, c.nt), 'obs_data')
    c = full_runtime_reduce(
        c, self_key='slf_src_amp_y_init', call_key="__call_src__"
    )
    c = full_runtime_reduce(c, self_key='self', call_key='__call__')
    check_shape(
        c.data.src_amp_y_init,
        (c.n_shots, c.src_per_shot, c.nt),
        'src_amp_y_init',
    )
    c = full_runtime_reduce(c, **c.plt.resolve, self_key='slf_plt')
    for k, v in c.plt.items():
        if k not in ['final', 'resolve', 'skip'] + c.plt.get('skip', []):
            plt.clf()
            v(data=c[f'data.{k}'].detach().cpu())

    if c.get('do_training', True):
        c.data.vp.requires_grad = False
        c.data.curr_src_amp_y = c.data.src_amp_y_init.clone()
        # c.data.curr_src_amp_y = torch.rand(*c.data.src_amp_y_init.shape).to(c.device)
        c.data.curr_src_amp_y.requires_grad = True

        c.train.opt = c.train.opt([c.data.curr_src_amp_y])
        loss_fn = torch.nn.MSELoss()

        capture_freq = c.train.n_epochs // c.train.num_captured_frames
        src_amp_frames = []
        obs_frames = []
        for epoch in range(c.train.n_epochs):
            if epoch % capture_freq == 0:
                src_amp_frames.append(
                    c.data.curr_src_amp_y.squeeze().detach().clone()
                )
            c.train.opt.zero_grad()

            num_calls = 0

            def closure():
                nonlocal num_calls
                num_calls += 1
                c.train.opt.zero_grad()
                out = dw.scalar(
                    c.data.vp,
                    c.dx,
                    c.dt,
                    source_amplitudes=c.data.curr_src_amp_y,
                    source_locations=c.data.src_loc_y,
                    receiver_locations=c.data.rec_loc_y,
                    pml_freq=c.freq,
                )
                loss = 1e6 * loss_fn(out[-1], c.data.obs_data)
                if num_calls == 1 and epoch % capture_freq == 0:
                    obs_frames.append(out[-1].squeeze().detach().cpu().clone())

                loss.backward()
                return loss

            loss = c.train.opt.step(closure)
            print(f'Epoch: {epoch}, Loss: {loss.item()}')
            if loss.item() < c.train.threshold:
                print('Threshold reached')
                break
            # torch.nn.utils.clip_grad_value_(
            #     c.data.curr_src_amp_y,
            #     torch.quantile(c.data.curr_src_amp_y.grad.detach().abs(), 0.98),
            # )
            # if loss.item() < c.train.threshold:
            #     break
            # c.train.opt.step()

        src_amp_frames.append(c.data.curr_src_amp_y.squeeze().detach().clone())
        src_amp_frames = torch.stack(src_amp_frames)
        obs_frames = torch.stack(obs_frames)

        def src_amp_plotter(*, data, idx, fig, axes):
            plt.clf()
            plt.title(idx)
            plt.plot(
                c.data.src_amp_y.squeeze().detach().cpu(),
                'r-',
                label='True Src Amp',
                markersize=3,
                alpha=0.75,
            )
            plt.plot(
                data[idx].detach().cpu(), 'b--', label='Curr Src Amp', lw=1
            )
            plt.legend()

        fig, axes = plt.subplots(1, 1)
        frames = get_frames_bool(
            data=src_amp_frames,
            iter=[(i, True) for i in range(src_amp_frames.shape[0])],
            fig=fig,
            axes=axes,
            plotter=src_amp_plotter,
        )
        save_frames(frames, path=hydra_out('history'))

        if c.save_tensors:
            torch.save(
                src_amp_frames.detach().cpu(), hydra_out('src_amp_frames.pt')
            )
            torch.save(
                c.data.src_amp_y.squeeze().detach().cpu(),
                hydra_out('true_src_amp_y.pt'),
            )
            os.system(
                'ln -s "$(pwd)/tmp_plotter.py" ' + hydra_out('tmp_plotter.py')
            )

            torch.save(obs_frames.detach().cpu(), hydra_out('obs_frames.pt'))
            torch.save(
                c.data.obs_data.squeeze().detach().cpu(),
                hydra_out('true_obs_data.pt'),
            )

        with open('.latest', 'w') as f:
            f.write(f'cd {hydra_out()}')

        print('Run following for latest run directory\n        . .latest')


if __name__ == "__main__":
    main()
