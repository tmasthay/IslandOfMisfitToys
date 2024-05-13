import hydra
from omegaconf import DictConfig


@hydra.main(config_path="cfg", config_name="cfg", version_base=None)
def main_dummy(cfg: DictConfig) -> None:
    print('hello')
    from main_worker import main

    main(cfg)


if __name__ == "__main__":
    main_dummy()
