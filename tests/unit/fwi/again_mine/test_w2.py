import pytest
from hydra import compose, initialize
from mh.core import DotDict
from omegaconf import DictConfig, OmegaConf


def preprocess_cfg(cfg):
    c = DotDict(OmegaConf.to_container(cfg, resolve=True))
    return c


def modify_cfg(cfg):
    return cfg


@pytest.fixture(scope='module')
def cfg():
    with initialize(config_path="cfg", job_name="test_w2", version_base=None):
        cfg = compose(config_name="cfg")
    c = preprocess_cfg(cfg)
    c = modify_cfg(c)
    return c


def test_cfg_read(cfg):
    assert len(list(cfg.keys())) != 0, "cfg is empty"


def test_unbatch_splines(cfg):
    assert False, "test_unbatch_splines not implemented"


def test_unbatch_splines_lambda(cfg):
    assert False, "test_unbatch_splines_lambda not implemented"


def test_pdf(cfg):
    assert False, "test_pdf not implemented"


def test_cdf(cfg):
    assert False, "test_cdf not implemented"


def test_disc_quantile(cfg):
    assert False, "test_disc_quantile not implemented"


def test_cts_quantile(cfg):
    assert False, "test_cts_quantile not implemented"


def test_get_quantile_lambda(cfg):
    assert False, "test_get_quantile_lambda not implemented"


def test_combos(cfg):
    assert False, "test_combos not implemented"
