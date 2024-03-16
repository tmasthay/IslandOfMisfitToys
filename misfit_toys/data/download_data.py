import sys

from misfit_toys.data.dataset import DataFactory


def download_data(storage, exclusions):
    DataFactory.create_database(storage=storage, exclusions=exclusions)

    print(
        "To see more examples, look at the source code located in"
        f" {__file__}.\n",
        "You can still access those examples directly through misfit_toys."
        "examples if you do not need the actual source code.",
    )


def main():
    storage = "conda/data"
    inclusions = set(sys.argv[1:])
    all = {'das_curtin', 'marmousi', 'marmousi2'}
    exclusions = list(all - inclusions)
    download_data(storage, exclusions)


if __name__ == "__main__":
    main()
