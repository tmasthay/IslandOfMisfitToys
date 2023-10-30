from masthay_helpers.global_helpers import save_metadata


@save_metadata(cli=True)
def metadata():
    return {
        'ext': 'npy',
        'num_urls': 10,
        'mode': 'front',
        'ny': 70,
        'nx': 70,
        'nt': 1000,
        'dt': 0.001,
        'n_shots': 5,
        'src_per_shot': 1,
        'd_src': 10,
        'fst_src': 5,
        'src_depth': 2,
        'rec_per_shot': 70,
        'd_rec': 1,
        'fst_rec': 0,
        'rec_depth': 2,
        'd_intra_shot': 0,
        'freq': 25,
        'peak_time': 1.5 / 25,
        'accuracy': 8,
        'derived': {'FlatVel_A': {}},
    }


if __name__ == "__main__":
    metadata()
