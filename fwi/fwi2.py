from deepwave_helpers import get_file, run_and_time
from deploy import *
import warnings

warnings.filterwarnings("ignore")
d = run_and_time('Start preprocess', 
    'End preprocess', 
    preprocess_data, 
    ny=2301,
    nx=751,
    n_shots=1,
    training={'n_epochs': 200}
)
d = run_and_time('Start training', 'End training', deploy_training, **d)
run_and_time('Start plotting', 'End plotting', postprocess, **d)