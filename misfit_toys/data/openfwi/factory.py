from subprocess import check_output as co
import os
from gdown import download
import numpy as np
import torch
import argparse

from masthay_helpers.global_helpers import add_root_package_path

curr_dir = os.path.dirname(__file__)
add_root_package_path(path=os.path.dirname(__file__), pkg="misfit_toys")
from misfit_toys.swiffer import sco
from misfit_toys.data.dataset import DataFactory
from misfit_toys.utils import parse_path


class Factory(DataFactory):
    def __extend_init__(self):
        self.hashes = Factory.get_hashes(
            os.path.join(self.src_path, "data.html")
        )
        self.urls = [
            f"https://drive.google.com/uc?id={hash}" for hash in self.hashes
        ]

    def _manufacture_data(self):
        for i, url in enumerate(self.urls):
            filename = f"{self.out_dir}/data_{i}"
            download(url, f"{filename}.npy", quiet=False)
            tensor = torch.from_numpy(np.load(f"{filename}.npy"))
            torch.save(tensor, f"{filename}.pt")
            os.remove(f"{filename}.npy")

    @staticmethod
    def get_hashes(filename):
        base_lines = f"cat {filename} | grep 'Storage used: ' | grep 'tooltip'"

        extract_first_cmd = base_lines + " | ".join(
            [
                "",
                "head -n 1",
                "grep -oP '<c-data.*</c-data>'",
                "awk -F';' '{print $(NF-1)}'",
            ]
        )

        extract_second_cmd = base_lines + " | ".join(
            [
                "",
                """grep -oP 'data-id=".{10,50}" class='""",
                """awk -F'"' '{print $2}'""",
            ]
        )

        u = [sco(extract_first_cmd, split=False)]
        u.extend(sco(extract_second_cmd, split=True))
        return u


def download_openfwi(*, storage, exclusions):
    Factory.create_database(storage=storage, exclusions=exclusions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--storage", default="conda/data")
    parser.add_argument("exclusions", nargs="*", default=[])
    args = parser.parse_args()
    args.storage = parse_path(args.storage)

    # input(args)

    download_openfwi(storage=args.storage, exclusions=args.exclusions)
