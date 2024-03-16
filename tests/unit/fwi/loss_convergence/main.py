import hydra
import torch
from mh.core import DotDict, convert_dictconfig, torch_stats, yamlfy
from mh.typlotlib import get_frames_bool, save_frames
from omegaconf import DictConfig

from misfit_toys.utils import apply, apply_all, bool_slice, resolve

torch.set_printoptions(precision=3, sci_mode=False, callback=torch_stats('all'))


def preprocess_cfg(cfg: DictConfig) -> DotDict:
    c = convert_dictconfig(cfg)
    c = resolve(c, relax=True)
    c = apply_all(c, relax=True)
    return c


def extend_cfg(c: DotDict) -> DotDict:
    for k, v in c.y.items():
        c.y[k].init_guess.requires_grad = True
        c[f'rt.{k}.obs_data'] = v.signal + v.noise
        c[f'rt.{k}.train.loss'] = c.train.loss(obs_data=c.rt[k].obs_data)
        c[f'rt.{k}.train.opt'] = c.train.opt([c.y[k].init_guess])
        c[f'rt.{k}.train.loss_history'] = torch.empty(c.train.max_iters + 1)
        c[f'rt.{k}.train.soln_history'] = torch.empty(
            c.train.max_iters + 1, *c.y[k].init_guess.shape
        )
        c[f'rt.{k}.train.grad_history'] = torch.empty(
            c.train.max_iters + 1, *c.y[k].init_guess.shape
        )

        soln = c.y[k].init_guess
        c.rt[k].train.loss_history[0] = c.rt[k].train.loss(soln)
        c.rt[k].train.soln_history[0] = soln.detach().cpu()
        c.rt[k].train.grad_history[0] = torch.zeros(soln.shape)
        for epoch in range(1, 1 + c.train.max_iters):
            c.rt[k].train.opt.zero_grad()
            loss = c.rt[k].train.loss(soln)
            loss.backward()
            c.rt[k].train.loss_history[epoch] = loss.detach().cpu()
            c.rt[k].train.soln_history[epoch] = soln.detach().cpu()
            c.rt[k].train.grad_history[epoch] = soln.grad.detach().cpu()
            if epoch % c.train.print_freq == 1:
                print(
                    f'Epoch: {epoch}, Loss: {loss}, Grad norm:'
                    f' {soln.grad.norm()}',
                    end='\r',
                )
            c.rt[k].train.opt.step()
        print()
        c.rt[k].train.grad_norm_history = c.rt[k].train.grad_history.norm(
            dim=-1
        )

    return c


def postprocess_cfg(c: DotDict) -> None:
    c.plt.callback(c)


@hydra.main(config_path="cfg", config_name="cfg", version_base=None)
def main(cfg):
    c = preprocess_cfg(cfg)
    c = extend_cfg(c)
    # print(c)
    postprocess_cfg(c)


if __name__ == "__main__":
    main()
