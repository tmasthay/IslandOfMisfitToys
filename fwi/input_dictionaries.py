from .deepwave_helpers import get_file, run_and_time
from .custom_losses import *
import numpy as np

def marmousi_section(**kw):
    return kw

def marmousi_full(**kw):
    d = {
        'ny': 2301,
        'nx': 751,
        'n_shots': 115,
        'n_receivers_per_shot': 384,
        'nt': 750
    }
    d = {**kw, **d}
    return d

def openfwi_layer_a(**kw):
    model = torch.from_numpy(np.load(get_file('model1.npy')))
    data = torch.from_numpy(np.load(get_file('data1.npy')))
    data = data.transpose(2,3)
    model = model[0,:,:]

    e = {
        'file_name': model[0],
        'obs_binary': data[0],
        'ny': 1000,
        'nx': 70,
        'nt': 1000,
        'dt': 0.01,
        'n_shots': 5,
        'n_receivers_per_shot': 70,
        'd_receiver': 0.01,
        'd_source': 0.175,
        'loss_fn': kw.get('loss_fn', torch.nn.MSELoss()),
        'v_init_lambda': lambda x : x,
        'training': {},
        'plotting': {}
    }

    e.update({'nx_full': e['nx'],
            'ny_full': e['ny'],
            'nt_full': e['nt'],
            'n_shots_full': e['n_shots'],
            'n_receivers_per_shot_full': e['n_receivers_per_shot'],
        }
    )
    e['training'].update({'n_epochs': 200, 'shots_per_batch': e['n_shots']})
    e['plotting'].update({ 
        'output_files': ['%s.jpg'%(str(e['loss_fn']).replace('()', ''))]
        }
    )
    return e