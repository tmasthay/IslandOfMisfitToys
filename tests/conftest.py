import os

import pytest
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from mh.core import DotDict


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
        return a + x * b

    return helper


@pytest.fixture(scope='session')
def lcl_cfg():
    def helper(cfg, key, inherit_keys=None):
        lcl_cfg = cfg[key]
        for k in inherit_keys:
            lcl_cfg[k] = lcl_cfg.get(k, cfg[k])
        return lcl_cfg

    return helper
