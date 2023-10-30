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
from misfit_toys.utils import parse_path, get_pydict


class Factory(DataFactory):
    def __extend_init__(self):
        data_urls = get_pydict(self.src_path, filename="data_urls")
        model_urls = get_pydict(self.src_path, filename="model_urls")
        self.metadata['data_urls'] = data_urls
        self.metadata['model_urls'] = model_urls

    def download_instance(self, k):
        input('yo')
        num_urls = self.metadata.get("num_urls", None)
        mode = self.metadata.get("mode", "front")
        curr_urls = self.metadata[f'{k}_urls']

        def choose(indices):
            key = lambda i: f'{k}{i}'
            return {key(i): curr_urls[key(i)] for i in indices}

        if num_urls is not None:
            if mode == "front":
                urls = choose(range(1, num_urls + 1))
            elif mode == "back":
                urls = choose(range(len(curr_urls) - num_urls, len(curr_urls)))
            elif mode == "random":
                urls = choose(np.random.choice(len(curr_urls), num_urls))

        for basename, url in urls.items():
            filename = os.path.join(self.out_path, basename)
            download(url, f"{filename}.npy", quiet=False)
            tensor = torch.from_numpy(np.load(f"{filename}.npy"))
            torch.save(tensor, f"{filename}.pt")
            os.remove(f"{filename}.npy")

    def _manufacture_data(self, **kw):
        self.download_instance('data')
        self.download_instance('model')

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


class FactorySignalOnly(DataFactory):
    def __extend__init(self):
        pass

    def _manufacture_data(self):
        pass


def signal_children():
    factory = FactorySignalOnly.cli_construct(
        device="cuda:0", src_path=os.path.dirname(__file__)
    )
    factory.manufacture_data()


if __name__ == "__main__":
    signal_children()
