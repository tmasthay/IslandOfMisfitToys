import os
from misfit_toys.data.dataset import DataFactory
import torch
import numpy as np


class Factory(DataFactory):
    def _manufacture_data(self):
        numpy_files = [
            os.path.join(self.out_path, e)
            for e in os.listdir(self.out_path)
            if e.endswith('.npy')
        ]
        for f in numpy_files:
            self.tensors[f.replace('.npy', '')] = torch.from_numpy(
                np.load(f)
            ).permute(0, 1, 3, 2)
            try:
                os.remove(f)
            except PermissionError:
                print(f'PermissionError: removal of {f}')


class FactorySignalOnly(DataFactory):
    def _manufacture_data(self):
        pass


def signal_children():
    factory = FactorySignalOnly.cli_construct(
        device="cuda:0", src_path=os.path.dirname(__file__)
    )
    factory.manufacture_data()


if __name__ == "__main__":
    signal_children()
