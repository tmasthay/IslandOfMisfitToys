import hydra
from omegaconf import DictConfig


# This is the main entry point for the application
#    The only reason for this restructuring is for faster autocompletion
#        since we bypass needing to import pytorch just to get the
#        autocompletion at command line.
@hydra.main(config_path="cfg", config_name="cfg", version_base=None)
def main_dummy(cfg: DictConfig) -> None:
    from main_worker import main

    main(cfg)


if __name__ == "__main__":
    main_dummy()
