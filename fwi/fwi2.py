from deepwave_helpers import get_file, run_and_time
from deploy import *
import warnings
from custom_losses import *
import numpy as np
from scipy.ndimage import gaussian_filter
from input_dictionaries import *
import torch

# e = openfwi_layer_a(loss_fn=W1())
# e = marmousi_full(optimiser_lambda=(lambda x : \
#         torch.optim.SGD([x], lr=0.1, momentum=0.9)
#     )
# )
e = marmousi_section()

warnings.filterwarnings("ignore")
d = run_and_time('Start preprocess', 'End preprocess', preprocess_data, **e)
d = run_and_time('Start training', 'End training', deploy_training, **d)
run_and_time('Start plotting', 'End plotting', postprocess, **d)
