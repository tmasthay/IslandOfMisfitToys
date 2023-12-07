from misfit_toys.data.dataset import DataFactory
from masthay_helpers.global_helpers import iprint

import os


def download_data(storage, exclusions):
    DataFactory.create_database(storage=storage, exclusions=exclusions)

    iprint(
        "To see more examples, look at the source code located in"
        f" {__file__}.\n",
        "You can still access those examples directly through misfit_toys."
        "examples if you do not need the actual source code.",
    )


def main():
    storage = "conda/data"
    exclusions = ["das_curtin"]
    download_data(storage, exclusions)


if __name__ == "__main__":
    main()
