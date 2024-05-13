import hydra
from omegaconf import DictConfig


@hydra.main(config_path="cfg", config_name="config", version_base=None)
def main_dummy(cfg: DictConfig) -> None:
    from main_worker import main

    main(cfg)
