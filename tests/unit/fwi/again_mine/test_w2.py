import pytest
from hydra import compose, initialize


@pytest.fixture(scope='module')
def cfg():
    with initialize(config_path="cfg", job_name="test_w2"):
        cfg = compose(config_name="cfg")
    return cfg


def test_cfg_read(cfg):
    assert cfg != {}, "cfg is empty"
