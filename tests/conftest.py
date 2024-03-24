import os
from copy import deepcopy

import pytest
import torch
import yaml
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from mh.core import DotDict
from omegaconf import OmegaConf

from misfit_toys.utils import apply_all, exec_imports


def load_hydra_config(config_dir='.', config_name='cfg'):
    # Only initialize GlobalHydra instance if it hasn't been initialized before
    if not GlobalHydra().is_initialized():
        initialize(config_path=config_dir, job_name="pytest_job")
        cfg = compose(config_name=config_name)
    else:
        # If already initialized, just compose the config
        cfg = compose(config_name=config_name)

    return DotDict(OmegaConf.to_container(cfg, resolve=True))


def special_preprocess_items(c: DotDict) -> DotDict:
    c.unit.beta.w2.x = torch.linspace(*c.unit.beta.w2.x)
    c.unit.beta.w2.p = torch.linspace(
        c.unit.beta.w2.eps, 1.0 - c.unit.beta.w2.eps, c.unit.beta.w2.np
    )
    c.unit.beta.conv.x = torch.linspace(*c.unit.beta.conv.x)
    c.unit.beta.conv.p = torch.linspace(
        c.unit.beta.conv.eps, 1.0 - c.unit.beta.conv.eps, c.unit.beta.conv.np
    )
    return c


def preprocess_cfg(cfg: DotDict) -> DotDict:
    c = special_preprocess_items(cfg)
    c = cfg.self_ref_resolve(gbl=globals(), lcl=locals())
    c = exec_imports(c)
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
