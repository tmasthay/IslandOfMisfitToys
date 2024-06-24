import inspect

import hydra
from omegaconf import DictConfig, OmegaConf

from misfit_toys.utils import self_read_cfg


@hydra.main(config_path="cfg", config_name="cfg", version_base=None)
def main(cfg: DictConfig):
    c = self_read_cfg(cfg)
    c.callback()


if __name__ == "__main__":
    main()
