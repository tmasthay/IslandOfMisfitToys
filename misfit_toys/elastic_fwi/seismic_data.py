import matplotlib.pyplot as plt
import torch
import os
from masthay_helpers.typlotlib import setup_gg_plot, rand_color, set_color_plot_global

from .elastic_class import *
from scipy.ndimage import gaussian_filter
from .elastic_custom import *

def marmousi_acoustic(): 
    devices = get_all_devices()
    vp_true = retrieve_dataset(
        field='vp',
        folder='marmousi',
        path=os.path.join(sco('echo $CONDA_PREFIX')[0], 'data')
    )
    ratio_y = 4
    ratio_x = 4
    vp_true = downsample_tensor(vp_true, axis=0, ratio=ratio_y)
    vp_true = downsample_tensor(vp_true, axis=1, ratio=ratio_x)
    # vp=torch.tensor(
    #     1./gaussian_filter(1./vp_true.numpy(), sigma=40.0)
    # ).to(devices[0])
    # vp = 3000.0 * torch.ones_like(vp_true).to(devices[0])
    mu = 0.0
    sig = 0.1
    vp = vp_true.clone()
    vp = (vp * (1.0 + mu + sig * torch.randn_like(vp))).to(devices[0])
    # vp=torch.tensor(
    #     1./gaussian_filter(1./vp.numpy(), sigma=40.0)
    # ).to(devices[0])
    vp.requires_grad=True

    uniform_survey = SurveyUniformLambda(
        n_shots=1,
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
            lambda *,self,pts,comp: None if pts == None else \
                deepwave.wavelets.ricker(
                    freq=self.custom['ricker_freq'],
                    length=self.nt,
                    dt=self.dt,
                    peak_time=self.custom['peak_time']
                ).repeat(*pts.shape[:-1], 1)
        ),
        deploy=[
            ('src_loc_y', devices[0]),
            ('src_amp_y', devices[0]),
            ('rec_loc_y', devices[0])
        ],
        ricker_freq=10.0,
        peak_time=0.05,
        nt=1000,
        dt=0.0001
    )

    model = Model(
        survey=uniform_survey,
        model='acoustic',
        vp=vp_true,
        u=None,
        rho=None,
        vs=None,
        freq=1.0,
        dy=0.004,
        dx=0.004,
        deploy=[('vp', devices[0])]
    )

    print('Computing observations...')
    obs_data = model.forward()
    model.vp = vp

    fwi_solver = FWI(
        obs_data=obs_data,
        model=model, 
        loss=torch.nn.MSELoss(reduction='sum'),
        optimizer=[
            torch.optim.SGD,
            {'lr': 1.0}
        ],
        scheduler=[
            (torch.optim.lr_scheduler.StepLR, {'step_size': 10, 'gamma': 0.9}),
            (torch.optim.lr_scheduler.ExponentialLR, {'gamma': 0.99})
        ],
        epochs=25,
        batch_size=1,
        trainable=['vp'],
        make_plots=[('vp', True)],
        print_freq=1,
        verbose=True,
        deploy=[],
        clip_grad=[('vp', 0.98)],
        loss_scaling=1.0e20
    )

    return fwi_solver, model, uniform_survey