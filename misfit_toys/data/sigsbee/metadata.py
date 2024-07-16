from mh.core_legacy import save_metadata


@save_metadata(cli=True)
def metadata():
    return {
        "url": "",
        "ext": "pt",
        "ny": 3201,
        "nx": 1201,
        "nt": 751,
        "dy": 0.00762,  # meters
        "dx": 0.00762,  # meters
        "dt": 0.004,  # seconds
        "n_shots": 50,
        "src_per_shot": 1,
        "d_src": 20,
        "fst_src": 10,
        "src_depth": 2,
        "rec_per_shot": 350,
        "d_rec": 3,
        "fst_rec": 0,
        "rec_depth": 2,
        "d_intra_shot": 0,
        "freq": 25,
        "peak_time": 1.5 / 25,
        "accuracy": 8,
        "vp_true": {"filename": "vstr2A"},
        "derived": {},
    }


if __name__ == "__main__":
    metadata()
