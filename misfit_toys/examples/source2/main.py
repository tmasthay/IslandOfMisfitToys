import deepwave as dw
import hydra
import matplotlib.pyplot as plt
import torch
from mh.core import DotDict, hydra_out, set_print_options, torch_stats
from mh.typlotlib import get_frames_bool, save_frames
from omegaconf import OmegaConf

from misfit_toys.utils import exec_imports, runtime_reduce

set_print_options(callback=torch_stats('all'))


@hydra.main(config_path="cfg", config_name="cfg", version_base=None)
def main(cfg):
    c = DotDict(OmegaConf.to_container(cfg, resolve=True))
    c = exec_imports(c)

    def runtime_reduce_simple(key):
        resolve_rules = c[key].get('resolve', c.resolve)
        self_key = f'slf_{key.split(".")[-1]}'
        c[key] = runtime_reduce(c[key], **resolve_rules, self_key=self_key)

    def full_runtime_reduce(lcl_cfg, **kw):
        return runtime_reduce(lcl_cfg, **{**lcl_cfg.resolve, **kw})

    runtime_reduce_simple('data.vp')
    runtime_reduce_simple('data.rec_loc_y')
    runtime_reduce_simple('data.src_loc_y')
    runtime_reduce_simple('data.src_amp_y')
    # runtime_reduce_simple('data.src_amp_y_init')
    runtime_reduce_simple('data.gbl_rec_loc')
    c = full_runtime_reduce(
        c, self_key='slf_gbl_obs_data', call_key="__call_gbl_obs__"
    )
    c = full_runtime_reduce(c, self_key='slf_obs_data', call_key="__call_obs__")
    c = full_runtime_reduce(
        c, self_key='slf_src_amp_y_init', call_key="__call_src__"
    )
    c = full_runtime_reduce(c, **c.plt.resolve, self_key='slf_plt')
    for k, v in c.plt.items():
        if k not in ['final', 'resolve', 'skip'] + c.plt.get('skip', []):
            plt.clf()
            v(data=c[f'data.{k}'].detach().cpu())

    if c.get('do_training', True):
        c.data.vp.requires_grad = False
        c.data.curr_src_amp_y = c.data.src_amp_y_init.clone()
        c.data.curr_src_amp_y.requires_grad = True
        # optimizer = torch.optim.LBFGS([c.data.curr_src_amp_y])
        optimizer = torch.optim.Adam([c.data.curr_src_amp_y], lr=c.train.lr)
        loss_fn = torch.nn.MSELoss()

        capture_freq = c.train.n_epochs // c.train.num_captured_frames
        src_amp_frames = []
        for epoch in range(c.train.n_epochs):
            if epoch % capture_freq == 0:
                src_amp_frames.append(
                    c.data.curr_src_amp_y.squeeze().detach().clone()
                )
            optimizer.zero_grad()

            def closure():
                optimizer.zero_grad()
                out = dw.scalar(
                    c.data.vp,
                    c.dx,
                    c.dt,
                    source_amplitudes=c.data.curr_src_amp_y,
                    source_locations=c.data.src_loc_y,
                    receiver_locations=c.data.rec_loc_y,
                    pml_freq=c.freq,
                )
                loss = loss_fn(out[-1], c.data.obs_data)
                loss.backward()
                return loss

            loss = optimizer.step(closure)
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
            # optimizer.step()

        src_amp_frames.append(c.data.curr_src_amp_y.squeeze().detach().clone())
        src_amp_frames = torch.stack(src_amp_frames)

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

        with open('.latest', 'w') as f:
            f.write(f'cd {hydra_out()}')

        print('Run following for latest run directory\n        . .latest')


if __name__ == "__main__":
    main()
