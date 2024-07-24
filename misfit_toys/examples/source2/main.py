import deepwave as dw
import hydra
import torch
from mh.core import DotDict, set_print_options, torch_stats
from omegaconf import OmegaConf

from misfit_toys.utils import exec_imports, runtime_reduce

set_print_options(callback=torch_stats('all'))


@hydra.main(config_path="cfg", config_name="cfg", version_base=None)
def main(cfg):
    c = DotDict(OmegaConf.to_container(cfg, resolve=True))
    c = exec_imports(c)

    def runtime_reduce_simple(key):
        resolve_rules = c[key].get('resolve', c.resolve)
        c[key] = runtime_reduce(c[key], **resolve_rules, self_key=f'slf_{key}')

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
    c = full_runtime_reduce(c, **c.plt.resolve, self_key='slf_plt')
    for k, v in c.plt.items():
        if k not in ['resolve', 'skip'] + c.plt.get('skip', []):
            v(data=c[f'data.{k}'].detach().cpu())

    if c.get('do_training', False):
        n_epochs = 250
        c.data.vp.requires_grad = False
        c.data.src_amp_y.requires_grad = True
        optimizer = torch.optim.SGD([c.data.src_amp_y], lr=0.1, momentum=0.9)
        loss_fn = torch.nn.MSELoss()

        for epoch in range(n_epochs):
            optimizer.zero_grad()
            out = dw.scalar(
                c.data.vp,
                c.dx,
                c.dt,
                source_amplitudes=c.data.src_amp_y,
                source_locations=c.data.src_loc_y,
                receiver_locations=c.data.rec_loc_y,
                pml_freq=c.freq,
            )
            loss = loss_fn(out[-1], c.data.obs_data)
            print(f'Epoch {epoch}: {loss.item()}')
            loss.backward()
            torch.nn.utils.clip_grad_value_(
                c.data.src_amp_y,
                torch.quantile(c.data.src_amp_y.grad.detach().abs(), 0.98),
            )
            optimizer.step()


if __name__ == "__main__":
    main()
