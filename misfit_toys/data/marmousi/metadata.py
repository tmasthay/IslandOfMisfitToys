from mh.core_legacy import save_metadata


@save_metadata(cli=True)
def metadata():
    return {
        "url": (
            "https://www.geoazur.fr/WIND/pub/nfs/FWI-DATA/GEOMODELS/Marmousi"
        ),
        "ext": "bin",
        "ny": 2301,
        "nx": 751,
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
        "accuracy": 8,
        "vp_true": {"filename": "vp"},
        "rho_true": {"filename": "rho"},
        "derived": {
            "deepwave_example": {
                "ny": 600,
                "nx": 250,
                "nt": 300,
                "n_shots": 20,
                "rec_per_shot": 100,
                "derived": {
                    "shots16": {
                        "n_shots": 16,
                        'derived': {
                            'twolayer': {'beta': 1.0},
                            'twolayer_strong': {'beta': 3.0},
                            'twolayer_verystrong': {'beta': 10.0},
                        },
                    }
                },
            }
        },
    }


if __name__ == "__main__":
    metadata()
