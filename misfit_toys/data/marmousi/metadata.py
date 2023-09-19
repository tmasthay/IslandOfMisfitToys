def metadata():
    return {
        'url': 'https://www.geoazur.fr/WIND/pub/nfs/FWI-DATA/' + 
            'GEOMODELS/Marmousi',
        'ext': 'bin',
        'ny': 2301,
        'nx': 751,
        'nt': 750,
        'dy': 4.0,
        'dx': 4.0,
        'dt': 0.004,
        'n_shots': 115,
        'src_per_shot': 1,
        'd_src': 20,
        'fst_src': 10,
        'src_depth': 2,
        'rec_per_shot': 384,
        'd_rec': 6,
        'fst_rec': 0,
        'rec_depth': 2,
        'd_intra_shot': 0,
        'freq': 25,
        'peak_time': 1.5 / 25,
        'accuracy': 8,
        'vp': {},
        'rho': {}
    }