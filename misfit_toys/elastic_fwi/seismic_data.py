import matplotlib.pyplot as plt
import torch
import os
import deepwave

from masthay_helpers.typlotlib import setup_gg_plot, rand_color, set_color_plot_global

from .elastic_class import *

def marmousi_acoustic():
    uniform_survey = SurveyUniformLambda(
        n_shots=1,
        fst_src=[[1, 1]],
        d_src=[[1, 2]],
        num_src=[[1, 1]],
        fst_rec=[[1, 2]],
        d_rec=[[1, 3]],
        num_rec=[[1, 100]],
        amp_func=lambda *,pts,comp: torch.ones(pts.shape)
    )

    model = Model(
        survey=uniform_survey,
        model='acoustic',
        vp=retrieve_dataset(
            field='vp', 
            folder='marmousi', 
            path=os.path.join(sco('echo $CONDA_PREFIX')[0], 'data')
        ),
        u=None,
        rho=None,
        vs=None,
        freq=1.0,
        dt=0.001,
        dy=0.004,
        dx=0.004
    )

    fwi_solver = FWI(
        model=model, 
        loss=torch.nn.MSELoss(),
        optimizer=[
            torch.optim.Adam,
            {'lr': 0.01}
        ],
        scheduler=[
            (torch.optim.lr_scheduler.StepLR, {'step_size': 10, 'gamma': 0.1}),
            (torch.optim.lr_scheduler.ExponentialLR, {'gamma': 0.99})
        ],
        epochs=100,
        batch_size=1,
        trainable=['vp']
    )

    return fwi_solver, model, uniform_survey