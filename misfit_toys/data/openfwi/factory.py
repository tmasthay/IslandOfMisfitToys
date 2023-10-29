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
        num_urls = self.metadata.get("num_urls", None)
        mode = self.metadata.get("mode", "front")
        if num_urls is not None:
            if mode == "front":
                self.urls = self.urls[:num_urls]
            elif mode == "back":
                self.urls = self.urls[-num_urls:]
            elif mode == "random":
                self.urls = np.random.choice(self.urls, size=num_urls)
        for i, url in enumerate(self.urls):
            filename = os.path.join(self.out_dir, f'data{i}')
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


def signal_children():
    factory = DataFactory.cli_construct(
        device="cuda:0", src_path=os.path.dirname(__file__)
    )
    factory.broadcast_meta()


if __name__ == "__main__":
    signal_children()
