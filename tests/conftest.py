import os
from copy import deepcopy
from datetime import datetime

import pytest
import torch
import yaml
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from mh.core import DotDict, draise, hydra_out, set_print_options, torch_stats
from omegaconf import OmegaConf

from misfit_toys.utils import apply_all, exec_imports, git_dump_info

set_print_options(precision=2, sci_mode=True, callback=torch_stats())


def load_hydra_config(config_dir='.', config_name='cfg'):
    # Only initialize GlobalHydra instance if it hasn't been initialized before
    if not GlobalHydra().is_initialized():
        initialize(config_path=config_dir, job_name="pytest_job")
        cfg = compose(config_name=config_name)
    else:
        # If already initialized, just compose the config
        cfg = compose(config_name=config_name)

    d = OmegaConf.to_container(cfg, resolve=True)
    root_path = os.path.dirname(__file__)
    day_folder = datetime.now().strftime('%d-%m-%Y')
    exact_time = datetime.now().strftime('%H-%M-%S')
    full_folder = os.path.join(root_path, 'outputs', day_folder, exact_time)
    full_folder = os.path.abspath(full_folder)
    os.makedirs(full_folder, exist_ok=True)
    s = yaml.dump(d)
    # s = s.replace('!!python/object:mh.core.DotDict', '')
    with open(os.path.join(full_folder, 'cfg.yaml'), 'w') as f:
        f.write(s)
    with open(os.path.join(full_folder, 'gitinfo.txt'), 'w') as f:
        f.write(git_dump_info())
    u = DotDict(d)
    u.hydra_out = full_folder
    return u


def special_preprocess_items(c: DotDict) -> DotDict:
    c.unit.beta.w2.x = torch.linspace(*c.unit.beta.w2.x)
    c.unit.beta.w2.p = torch.linspace(
        c.unit.beta.w2.p.eps, 1.0 - c.unit.beta.w2.p.eps, c.unit.beta.w2.p.np
    )
    c.unit.beta.conv.x = torch.linspace(
        *c.unit.beta.conv.x.specs, device=c.unit.beta.conv.x.device
    )
    c.unit.beta.conv.p = torch.linspace(
        c.unit.beta.conv.p.eps,
        1.0 - c.unit.beta.conv.p.eps,
        c.unit.beta.conv.p.np,
        device=c.unit.beta.conv.p.device,
    )
    return c


def preprocess_cfg(cfg: DotDict) -> DotDict:
    # c = special_preprocess_items(cfg)
    # c = cfg.self_ref_resolve(gbl=globals(), lcl=locals())
    # c = exec_imports(c)
    # c = apply_all(c, relax=False)
    # c = c.self_ref_resolve(gbl=globals(), lcl=locals(), relax=False)
    # draise(c)
    # return c

    # os.makedirs(os.path.join(full_folder, 'figs'), exist_ok=True)

    c = exec_imports(cfg)
    c = special_preprocess_items(c)
    c = c.self_ref_resolve(gbl=globals(), lcl=locals(), relax=True)
    c = apply_all(c, relax=True)
    c = c.self_ref_resolve(gbl=globals(), lcl=locals(), relax=False)
    c = apply_all(c, relax=False)
    return c


@pytest.fixture(scope='session')
def cfg():
    c = load_hydra_config()
    c = preprocess_cfg(c)
    return c


@pytest.fixture(scope='session')
def adjust():
    def helper(x, a, b):
        return a + x * (b - a)

    return helper


@pytest.fixture(scope='session')
def lcl_cfg():
    def helper(cfg, key, inherit_keys=None):
        lcl_cfg = deepcopy(cfg[key])
        for k in inherit_keys:
            lcl_cfg[k] = lcl_cfg.get(k, cfg[k])
        return lcl_cfg

    return helper


@pytest.fixture(scope='session')
def report_cfg(cfg):
    def helper(c, name):
        if cfg.verbose >= 2:
            s = yaml.dump(c.__dict__)
            s = s.replace("!!python/object:mh.core.DotDict", "")
            print(f'\n\n{name}\n\n{s}\n\n', flush=True)

    return helper
