import os
from misfit_toys.data.openfwi import Factory


class FactoryNpyToTorch(Factory):
    def __extend_init__(self):
        super().__extend_init__()
        u = [
            e
            for e in os.listdir(self.out_path)
            if e.endswith('.pt') or e.endswith('.npy')
        ]
        assert len(u) >= 120, f"Only {len(u)} files in {self.out_path}"


def main():
    f = FactoryNpyToTorch.cli_construct(
        device="cuda:0", src_path=os.path.dirname(__file__)
    )
    f.manufacture_data()


if __name__ == "__main__":
    main()
