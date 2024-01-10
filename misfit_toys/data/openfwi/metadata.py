from masthay_helpers.global_helpers import save_metadata
import os


@save_metadata(cli=True)
def metadata():
    file_path = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(file_path, 'api_key.secret'), 'r') as f:
        api_key = f.read().strip()

    return {
        'ext': 'npy',
        'num_files': 60,
        'api_key': api_key,
        'chunk_size': 1024,
        'data_size': int(667.7 * 1024.0**2),
        'model_size': int(9.4 * 1024**2),
        'static_file_size': True,
        'mode': 'front',
        'ny': 70,
        'nx': 70,
        'nt': 1000,
        'dt': 0.001,
        'dy': 10.0,
        'dx': 10.0,
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
        'derived': {
            'FlatVel_A': {
                'data_folder_id': '1arNrV9M65cl70ANkBwkg7bi7SI5JtsYQ',
                'model_folder_id': '1IBM_04bPCBnSMO1TMdaps3EHV5dYhYJa',
            },
            'CurveVel_A': {
                'data_folder_id': '1ry69BNdgG4eTIkKGbnTiYCVhjMKo0THw',
                'model_folder_id': '1AL7z9TQVgzUdv3yrHB6f_wP2TRvVTKD0',
            },
        },
    }


if __name__ == "__main__":
    metadata()
