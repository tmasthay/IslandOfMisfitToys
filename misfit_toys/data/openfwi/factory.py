from subprocess import check_output as co
import os
from gdown import download
import numpy as np
import torch
import argparse

from masthay_helpers.global_helpers import add_root_package_path, DotDict

curr_dir = os.path.dirname(__file__)
add_root_package_path(path=os.path.dirname(__file__), pkg="misfit_toys")
from misfit_toys.swiffer import sco
from misfit_toys.data.dataset import DataFactory, towed_src, fixed_rec
from misfit_toys.utils import parse_path, get_pydict
from scipy.ndimage import gaussian_filter
import deepwave as dw
from time import sleep


class Factory(DataFactory):
    def __extend_init__(self):
        data_urls = get_pydict(self.src_path, filename="data_urls")
        model_urls = get_pydict(self.src_path, filename="model_urls")
        self.metadata['data_urls'] = data_urls
        self.metadata['model_urls'] = model_urls

    def _manufacture_data(self):
        if self.installed(
            "vp_true",
            "vp_init",
            "rho_true",
            "src_loc_y",
            "rec_loc_y",
            "obs_data",
        ):
            return

        self.download_all()
        d = DotDict(self.metadata)
        self.tensors.vp_true = torch.load(
            os.path.join(self.out_path, 'model1.pt')
        )[0].squeeze()
        self.tensors.vp = self.tensors.vp_true.to(self.device)
        # self.tensors.vp_init = torch.tensor(
        #     1 / gaussian_filter(1 / self.tensors.vp_true.cpu().numpy(), 40)
        # )
        self.tensors.vp_init = self.tensors.vp_true.mean() * torch.ones_like(
            self.tensors.vp_true
        )
        d.ny, d.nx = self.tensors.vp_true.shape
        self.tensors.src_loc_y = towed_src(
            n_shots=d.n_shots,
            src_per_shot=d.src_per_shot,
            d_src=d.d_src,
            fst_src=d.fst_src,
            src_depth=d.src_depth,
            d_intra_shot=d.d_intra_shot,
        ).to(self.device)
        self.tensors.rec_loc_y = fixed_rec(
            n_shots=d.n_shots,
            rec_per_shot=d.rec_per_shot,
            d_rec=d.d_rec,
            fst_rec=d.fst_rec,
            rec_depth=d.rec_depth,
        ).to(self.device)
        self.tensors.src_amp_y = (
            dw.wavelets.ricker(d.freq, d.nt, d.dt, d.peak_time)
            .repeat(d.n_shots, d.src_per_shot, 1)
            .to(self.device)
        )
        print(f"Building obs_data in {self.out_path}...", end="", flush=True)
        self.tensors.obs_data = dw.scalar(
            self.tensors.vp,
            d.dy,
            d.dt,
            source_amplitudes=self.tensors.src_amp_y,
            source_locations=self.tensors.src_loc_y,
            receiver_locations=self.tensors.rec_loc_y,
            pml_freq=d.freq,
            accuracy=d.accuracy,
        )[-1]
        print("SUCCESS", flush=True)

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
            try:
                filename = os.path.join(self.out_path, basename)
                download(url, f"{filename}.npy", quiet=False)
                tensor = torch.from_numpy(np.load(f"{filename}.npy"))
                torch.save(tensor, f"{filename}.pt")
                os.remove(f"{filename}.npy")
            except:
                break

    def download_all(self):
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
