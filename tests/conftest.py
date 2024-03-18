import os

import pytest
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra


def load_hydra_config(config_dir='.', config_name='cfg'):
    # Only initialize GlobalHydra instance if it hasn't been initialized before
    if not GlobalHydra().is_initialized():
        initialize(config_path=config_dir, job_name="pytest_job")
        cfg = compose(config_name=config_name)
    else:
        # If already initialized, just compose the config
        cfg = compose(config_name=config_name)

    return cfg


@pytest.fixture(scope='session')
def cfg():
    return load_hydra_config()
