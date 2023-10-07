from ddp import ExampleIOMT
from ddp_deepwave import MultiscaleExample
from misfit_toys.examples.example import ExampleComparator


def main():
    common_tensors_names = [
        "vp_true",
        "vp_init",
        "vp_record",
        "freqs",
        "loss",
        "src_amp_y",
        "src_loc_y",
        "rec_loc_y",
        "obs_data",
    ]
    verbosity = 2
    iomt_example = ExampleIOMT(
        data_save="iomt/data",
        fig_save="iomt/figs",
        tensor_names=common_tensors_names,
        verbose=verbosity,
    )
    deepwave_example = MultiscaleExample(
        data_save="dw/data",
        fig_save="dw/figs",
        pickle_save="dw/pickle",
        verbose=verbosity,
        tensor_names=common_tensors_names,
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
