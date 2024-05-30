"""
This example is a copy of ``misfit_toys.hydra.main`` but in an isolated directory rather
than inside the package itself.

See also:
    - Documentation for ``misfit_toys.hydra.main``.
    - Hydra documentation: https://hydra.cc/docs/intro/
"""

import hydra
from omegaconf import DictConfig


@hydra.main(config_path="cfg", config_name="cfg", version_base=None)
def main_dummy(cfg: DictConfig) -> None:
    if __name__ != "__main__":
        raise RuntimeError(
            "This function should be run only as a main program from the"
            " command line."
        )

    from misfit_toys.hydra.main_worker import main

    main(cfg)


if __name__ == "__main__":
    main_dummy()
