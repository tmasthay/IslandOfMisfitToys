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

    def download_instance(self, k, indices='all'):
        prev_res = [
            e
            for e in os.listdir(self.out_path)
            if e.endswith('.pt') and e.startswith(k)
        ]
        if len(prev_res) > 0:
            print(
                f'Already downloaded {len(prev_res)} files in'
                f' {self.out_path}...skipping'
            )
            return
        urls = self.metadata[f'{k}_urls']
        if indices is not None:
            urls = {f'{k}{i+1}': urls[f'{k}{i+1}'] for i in indices}

        for basename, url in urls.items():
            filename = os.path.join(self.out_path, basename)
            download(url, f"{filename}.npy", quiet=False)
            tensor = torch.from_numpy(np.load(f"{filename}.npy"))
            torch.save(tensor, f"{filename}.pt")
            os.remove(f"{filename}.npy")

    def _manufacture_data(self):
        num_urls = self.metadata.get("num_urls", None)
        mode = self.metadata.get("mode", "front")
        N = len(self.metadata['data_urls'].keys())
        M = len(self.metadata['model_urls'].keys())
        assert N == M, 'data and model urls must be the same size'
        if mode == 'front':
            indices = range(num_urls)
        elif mode == 'back':
            indices = range(N - num_urls, N)
        elif mode == 'random':
            indices = np.random.choice(range(N), size=num_urls)
        else:
            raise ValueError(f'Invalid mode: {mode}')
        self.download_instance('data', indices)
        self.download_instance('model', indices)

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
