import os
from masthay_helpers.global_helpers import add_root_package_path

add_root_package_path(path=os.path.dirname(__file__), pkg='misfit_toys')
from misfit_toys.data.dataset import DataFactory, towed_src, fixed_rec
from misfit_toys.utils import DotDict


class Factory(DataFactory):
    def _manufacture_data(self):
        if self.installed('geophone_curtin', 'das_curtin'):
            self.tensors = DotDict({})
            return None

        d = DotDict(self.process_web_data())
        return d


def main():
    f = Factory.cli_construct(
        device='cuda:0', src_path=os.path.dirname(__file__)
    )
    f.manufacture_data()


if __name__ == "__main__":
    main()
