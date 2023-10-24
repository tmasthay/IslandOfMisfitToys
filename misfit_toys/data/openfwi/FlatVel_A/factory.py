import os
import argparse

from masthay_helpers.global_helpers import add_root_package_path

add_root_package_path(path=os.path.dirname(__file__), pkg="misfit_toys")
from misfit_toys.data.openfwi import Factory


def main():
    f = Factory.cli_construct(
        device="cuda:0", src_path=os.path.dirname(__file__)
    )
    f.manufacture_data()


if __name__ == "__main__":
    main()
