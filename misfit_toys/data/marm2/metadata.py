from mh.core_legacy import save_metadata


@save_metadata(cli=True)
def metadata():
    return {
        "url": "http://www.agl.uh.edu/downloads/",
        "ext": "segy",
        "unzip": True,
        "ny": 13601,
        "nx": 2801,
        "nt": 750,
        "dy": 4.0,
        "dx": 4.0,
        "dt": 0.004,
        "n_shots": 115,
        "src_per_shot": 1,
        "d_src": 20,
        "fst_src": 10,
        "src_depth": 2,
        "rec_per_shot": 384,
        "d_rec": 6,
        "fst_rec": 0,
        "rec_depth": 2,
        "d_intra_shot": 0,
        "freq": 25,
        "peak_time": 1.5 / 25,
        "accuracy": 4,
        "vp_true": {"filename": "vp_marmousi-ii.segy.gz"},
        "vs_true": {"filename": "vs_marmousi-ii.segy.gz"},
        "rho_true": {"filename": "density_marmousi-ii.segy.gz"},
        "derived": {
            "tiny": {
                "ny": 600,
                "nx": 250,
                "nt": 300,
                "n_shots": 20,
                "rec_per_shot": 100,
            },
            "medium": {"ny": 2301, "nx": 751},
        },
    }


if __name__ == "__main__":
    metadata()
