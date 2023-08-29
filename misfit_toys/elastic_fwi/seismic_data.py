import matplotlib.pyplot as plt
import torch
import os
from masthay_helpers.typlotlib import setup_gg_plot, rand_color, set_color_plot_global

from .elastic_class import *
from scipy.ndimage import gaussian_filter

def marmousi_acoustic():
    devices = get_all_devices()
    vp_true = retrieve_dataset(
        field='vp',
        folder='marmousi',
        path=os.path.join(sco('echo $CONDA_PREFIX')[0], 'data')
    )
    vp=torch.tensor(
        1./gaussian_filter(1./vp_true.numpy(), sigma=40.0)
    ).to(devices[0])
    vp.requires_grad=True

    uniform_survey = SurveyUniformLambda(
        n_shots=115,
        src_y={
            'src_per_shot': 1,
            'fst_src': 1,
            'src_depth': 2,
            'd_src': 20,
            'd_intra_shot': 0
        },
        rec_y={
            'rec_per_shot': 1,
            'fst_rec': 1,
            'rec_depth': 2,
            'd_rec': 20,
        },
        amp_func=(
            lambda *,self,pts,comp: \
                deepwave.ricker.wavelet(
                    freq=self.custom['ricker_freq'],
                    nt=self.nt,
                    dt=self.dt,
                    peak_time=self.custom['peak_time']
                ).repeat(*pts.shape, 1)
        ),
        deploy=[
            ('src_loc_y', devices[0]),
            ('src_amp_y', devices[0]),
            ('rec_loc_y', devices[0])
        ],
        ricker_freq=1.0,
        peak_time=0.08
    )

    model = Model(
        survey=uniform_survey,
        model='acoustic',
        vp=vp_true,
        u=None,
        rho=None,
        vs=None,
        freq=1.0,
        dt=0.001,
        dy=0.004,
        dx=0.004,
        deploy=[('vp', devices[0])]
    )

    obs_data = model.forward()
    model.vp = vp

    fwi_solver = FWI(
        obs_data=obs_data,
        model=model, 
        loss=torch.nn.MSELoss(),
        optimizer=[
            torch.optim.Adam,
            {'lr': 0.1}
        ],
        scheduler=[
            (torch.optim.lr_scheduler.StepLR, {'step_size': 10, 'gamma': 0.1}),
            (torch.optim.lr_scheduler.ExponentialLR, {'gamma': 0.99})
        ],
        epochs=10,
        batch_size=1,
        trainable=['vp'],
        make_plots=[('vp', True)],
        print_freq=1,
        verbose=True,
        deploy=[]
    )

    return fwi_solver, model, uniform_survey