import torch
from masthay_helpers.global_helpers import hydra_kw, DotDict
from omegaconf import DictConfig
import os
import pickle
from scipy.special import erfinv


def gen_gauss(cfg: DotDict):
    x = torch.linspace(*cfg.support, cfg.nx).expand(1, 1, cfg.nx)
    shifts = (
        torch.linspace(-cfg.shift, cfg.shift, cfg.nshifts)
        .unsqueeze(0)
        .unsqueeze(-1)
    )
    scales = (
        torch.linspace(*cfg.scales, cfg.nscales).unsqueeze(-1).unsqueeze(-1)
    )
    eps = 1e-3
    gauss = torch.exp(-((x - cfg.mu - shifts) ** 2 / (2 * scales**2))) + eps
    gauss /= (
        torch.trapezoid(gauss, x=x, dim=-1).unsqueeze(-1).expand(*gauss.shape)
    )
    # input(x.expand(*gauss.shape).shape)
    # input(f'gauss.shape = {gauss.shape}')
    ref = gauss[cfg.nscales // 2, cfg.nshifts // 2, :]
    expected_output = (shifts - cfg.mu) ** 2 + (scales - cfg.sigma) ** 2
    return DotDict(
        {
            'x': x.reshape(-1),
            'ref': ref,
            'gauss': gauss,
            'expected_output': expected_output,
            'mu': cfg.mu - shifts,
            'sigma': scales,
            'p': torch.linspace(0, 1, cfg.num_probs),
        }
    )


@hydra_kw(use_cfg=True)
def gen_data(cfg: DictConfig):
    cfg = DotDict(cfg.__dict__['_content'])
    filename = cfg.glob.filename.replace('.pkl', '') + '.pkl'

    gen_new = not os.path.exists(filename)
    if not gen_new:
        pickle_modify_time = os.stat(filename).st_mtime
        dependency_modify_time = min(
            map(lambda x: os.stat(x).st_mtime, [__file__, 'config.yaml'])
        )
        gen_new = pickle_modify_time < dependency_modify_time
    if gen_new:
        print(f'Generating {filename}')
        d = DotDict({'gauss': gen_gauss(cfg.gauss)})
        with open(cfg.glob.filename.replace('.pkl', '') + '.pkl', 'wb') as f:
            pickle.dump(d, f)
        return d
    else:
        with open(filename, 'rb') as f:
            return pickle.load(f)
