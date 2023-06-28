from deepwave_helpers import get_file, run_and_time
from deploy import *
import warnings
from custom_losses import *
import numpy as np
from scipy.ndimage import gaussian_filter
from input_dictionaries import *

e = openfwi_layer_a()

warnings.filterwarnings("ignore")
d = run_and_time('Start preprocess', 'End preprocess', preprocess_data, **e)
d = run_and_time('Start training', 'End training', deploy_training, **d)
run_and_time('Start plotting', 'End plotting', postprocess, **d)