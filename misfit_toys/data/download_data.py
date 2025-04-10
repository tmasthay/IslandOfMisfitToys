import sys

from misfit_toys.data.dataset import DataFactory


def download_data(storage, *, inclusions):
    all = {'das_curtin', 'marmousi', 'marm2'}
    exclusions = list(all - set(inclusions))
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
    # print(f'{inclusions=}')
    # all = {'das_curtin', 'marmousi', 'marmousi2'}
    # exclusions = list(all - inclusions)
    download_data(storage, inclusions=inclusions)


if __name__ == "__main__":
    main()
