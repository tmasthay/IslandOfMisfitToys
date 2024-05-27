from masthay_helpers.global_helpers import iprint

import misfit_toys
from misfit_toys.data.dataset import DataFactory


def download_data(storage, exclusions):
    DataFactory.create_database(storage=storage, exclusions=exclusions)


if __name__ == "__main__":
    print("Redownloading data...", end="")
    storage = "conda/data"
    exclusions = ["das_curtin"]
    download_data(storage, exclusions)
    print("SUCCESS")
