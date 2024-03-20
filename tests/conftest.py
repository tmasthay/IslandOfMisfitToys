import os

import pytest
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from mh.core import DotDict
import yaml
from copy import deepcopy


def load_hydra_config(config_dir='.', config_name='cfg'):
    # Only initialize GlobalHydra instance if it hasn't been initialized before
    if not GlobalHydra().is_initialized():
        initialize(config_path=config_dir, job_name="pytest_job")
        cfg = compose(config_name=config_name)
    else:
        # If already initialized, just compose the config
        cfg = compose(config_name=config_name)

    return DotDict(OmegaConf.to_container(cfg, resolve=True))


@pytest.fixture(scope='session')
def cfg():
    return load_hydra_config()


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
        if cfg.verbose >= 1:
            s = yaml.dump(c.__dict__)
            s = s.replace("!!python/object:mh.core.DotDict", "")
            print(f'\n\n{name}\n\n{s}\n\n', flush=True)

    return helper
