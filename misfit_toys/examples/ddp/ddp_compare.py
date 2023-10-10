from .ddp_driver import ExampleIOMT
from .ddp_deepwave import MultiscaleExample
from misfit_toys.examples.example import ExampleComparator
from misfit_toys.utils import canonical_reduce


def main():
    reduce = canonical_reduce(
        exclude=["src_amp_y", "rec_loc_y", "src_loc_y"],
        extra=["vp_record", "vp_true", "vp_init"],
    )
    verbosity = 2
    iomt_example = ExampleIOMT(
        data_save="iomt/data",
        fig_save="iomt/figs",
        reduce=reduce,
        verbose=verbosity,
    )
    deepwave_example = MultiscaleExample(
        data_save="dw/data",
        fig_save="dw/figs",
        pickle_save="dw/pickle",
        verbose=verbosity,
        reduce=reduce,
    )
    deepwave_example.n_epochs = 2

    cmp = ExampleComparator(
        iomt_example,
        deepwave_example,
        protect=["freqs"],
    )

    cmp.compare()


if __name__ == "__main__":
    main()
