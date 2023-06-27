from deepwave_helpers import get_file, run_and_time
from deploy import *
import warnings
from custom_losses import *

warnings.filterwarnings("ignore")
d = run_and_time('Start preprocess', 
    'End preprocess', 
    preprocess_data, 
    ny=600,
    nx=250,
    n_shots=30,
    training={'n_epochs': 250, 'shots_per_batch': 30},
    plotting={'output_files': ['L2.jpg']},
    loss_fn=torch.nn.MSELoss()
)
d = run_and_time('Start training', 'End training', deploy_training, **d)
run_and_time('Start plotting', 'End plotting', postprocess, **d)