import os

# add_root_package_path(path=os.path.dirname(__file__), pkg="misfit_toys")
from mh.core import DotDict

from misfit_toys.data.dataset import DataFactory, fixed_rec, towed_src


class Factory(DataFactory):
    def _manufacture_data(self):
        if self.installed("geophone_curtin", "das_curtin"):
            self.tensors = DotDict({})
            return None

        d = DotDict(self.process_web_data())
        return d


def main():
    f = Factory.cli_construct(
        device="cuda:0", src_path=os.path.dirname(__file__)
    )
    f.manufacture_data()


if __name__ == "__main__":
    main()
