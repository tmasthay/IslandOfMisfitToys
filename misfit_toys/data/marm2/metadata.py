from mh.core_legacy import save_metadata


def update_downample_params(v, *, ref):
    v['dy'] = ref['dy'] * v['down_y']
    v['dx'] = ref['dx'] * v['down_x']
    v['dt'] = ref['dt'] * v['down_t']
    v['ny'] = ref['ny'] // v['down_y']
    v['nx'] = ref['nx'] // v['down_x']
    v['nt'] = ref['nt'] // v['down_t']
    v['n_shots'] = ref['n_shots'] // v['down_shots']
    v['rec_per_shot'] = ref['rec_per_shot'] // v['down_rps']
    v['src_per_shot'] = ref['src_per_shot'] // v['down_sps']
    return v


@save_metadata(cli=True)
def metadata():
    d = {
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
        "d_src": 110,
        "fst_src": 10,
        "src_depth": 2,
        "rec_per_shot": 384,
        "d_rec": 32,
        "fst_rec": 1,
        "rec_depth": 10,
        "d_intra_shot": 0,
        "freq": 25,
        "peak_time": 1.5 / 25,
        "accuracy": 4,
        "vp_true": {"filename": "vp_marmousi-ii.segy.gz"},
        "vs_true": {"filename": "vs_marmousi-ii.segy.gz"},
        "rho_true": {"filename": "density_marmousi-ii.segy.gz"},
        "derived": {
            # "tiny": {
            #     "ny": 600,
            #     "nx": 250,
            #     "nt": 300,
            #     "n_shots": 20,
            #     "rec_per_shot": 100,
            # },
            # "medium": {"ny": 2301, "nx": 751},
            "tiny": {},
            "medium": {},
            "smooth": {
                "down_y": 4,
                "down_x": 4,
                "down_t": 1,
                "down_shots": 2,
                "down_rps": 1,
                "down_sps": 1,
                "delta": 10
            },
            "super_smooth": {
                "down_y": 16,
                "down_x": 16,
                "down_t": 1,
                "down_shots": 2,
                "down_rps": 1,
                "down_sps": 1,
                "delta": 10
            },
        },
    }
    d['derived']['smooth'] = update_downample_params(
        d['derived']['smooth'], ref=d
    )
    d['derived']['super_smooth'] = update_downample_params(
        d['derived']['super_smooth'], ref=d
    )
    return d


if __name__ == "__main__":
    metadata()
