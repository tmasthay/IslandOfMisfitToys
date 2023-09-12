import matplotlib.pyplot as plt
import torch
import os
from masthay_helpers.typlotlib import setup_gg_plot, rand_color, set_color_plot_global

from .elastic_class import *
from scipy.ndimage import gaussian_filter
from deepwave import scalar
from ..base_helpers import *


def marmousi_acoustic2():
    device = torch.device('cuda')
    path = os.path.join(sco('echo $CONDA_PREFIX')[0], 'data')
    vp_true = retrieve_dataset(
        field='vp',
        folder='marmousi',
        path=path
    )
    ny = 2301
    nx = 751
    dy = 4.0
    dx = 4.0
    # v_init = torch.tensor(1/gaussian_filter(1/vp_true.numpy(), 10))
    v_init = 4500 * torch.ones_like(vp_true)

    freq = 25
    nt = 750
    dt = 0.004
    peak_time = 1.5 / freq

    n_shots = 115

    def taper(x):
        return deepwave.common.cosine_taper_end(x, 100)

    # Select portion of data for inversion
    n_receivers_per_shot = 100
    nt = 300

    data_path = os.path.join(path, 'marmousi_data.pt')
    if os.path.exists(data_path):
        print('Loading Marmousi data directly...')
        observed_data = torch.load(data_path)
    else:
        n_shots = 115

        n_sources_per_shot = 1
        d_source = 20  # 20 * 4m = 80m
        first_source = 10  # 10 * 4m = 40m
        source_depth = 2  # 2 * 4m = 8m

        n_receivers_per_shot = 384
        d_receiver = 6  # 6 * 4m = 24m
        first_receiver = 0  # 0 * 4m = 0m
        receiver_depth = 2  # 2 * 4m = 8m
        # source_locations
        source_locations = torch.zeros(n_shots, n_sources_per_shot, 2,
                                    dtype=torch.long)
        source_locations[..., 1] = source_depth
        source_locations[:, 0, 0] = (torch.arange(n_shots) * d_source +
                                    first_source)

        # receiver_locations
        receiver_locations = torch.zeros(n_shots, n_receivers_per_shot, 2,
                                        dtype=torch.long)
        receiver_locations[..., 1] = receiver_depth
        receiver_locations[:, :, 0] = (
            (torch.arange(n_receivers_per_shot) * d_receiver +
            first_receiver)
            .repeat(n_shots, 1)
        )

        # source_amplitudes
        source_amplitudes = (
            (deepwave.wavelets.ricker(freq, nt, dt, peak_time))
            .repeat(n_shots, n_sources_per_shot, 1)
        )
        print('Generating Marmousi data...')
        observed_data = (
            scalar(
                vp_true[:ny, :nx].to(device), dx, dt,
                source_amplitudes=source_amplitudes.to(device),
                source_locations=source_locations.to(device),
                receiver_locations=receiver_locations.to(device),
                pml_freq=freq
            )[-1]
            .to('cpu')
        )
        torch.save(observed_data, data_path)
        observed_data = observed_data
        print(f'Data generated...run again!')
        exit(-1)

    observed_data = taper(observed_data[:n_shots, :n_receivers_per_shot, :nt])
    observed_data = observed_data.to(device)

    devices = get_all_devices()
    vp_true = vp_true[:ny, :nx]
    v_init = v_init[:ny, :nx]

    ricker_freq = 25.0
    n_shots_train = 4
    uniform_survey = SurveyUniformLambda(
        n_shots=n_shots,
        src_y={
            'src_per_shot': 1,
            'fst_src': 1,
            'src_depth': 2,
            'd_src': 20,
            'd_intra_shot': 0
        },
        rec_y={
            'rec_per_shot': 100,
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
        y_amp_param=Param,
        x_amp_param=Param,
        ricker_freq=ricker_freq,
        peak_time=peak_time,
        nt=nt,
        dt=dt
    )

    pml_freq = ricker_freq
    model = Model(
        survey=uniform_survey,
        model='acoustic',
        vp=v_init,
        rho=None,
        vs=None,
        vp_param=Param,
        vs_param=Param,
        rho_param=Param,
        u=None,
        freq=pml_freq,
        dy=dy,
        dx=dx
    )

    prop = Prop(
        model=model,
        train={
            'vp': True,
            'rho': False,
            'vs': False,
            'src_amp_y': False,
            'src_amp_x': False
        },
        device=devices[0]
    )

    fwi_solver = FWI(
        prop=prop,
        obs_data=observed_data,
        loss=torch.nn.MSELoss(),
        optimizer=[torch.optim.SGD, {'lr': 1.0}],
        scheduler=[
            (
                torch.optim.lr_scheduler.StepLR,
                {'step_size': 10, 'gamma': 0.9}
            ),
            (
                torch.optim.lr_scheduler.ExponentialLR,
                {'gamma': 0.99}
            )
        ],
        epochs=5,
        num_batches=n_shots // 5,
        multi_gpu=False,
        verbosity='progress',
        protocol=print,
        make_plots=[('vp', True)]
    )
    
    print(
        see_fields(
            fwi_solver,
            field='device',
            member_paths=[
                'obs_data',
                'prop.model.vp.param',
                'prop.model.vs.param',
                'prop.model.rho.param',
                'prop.model.survey.src_amp_y.param',
                'prop.model.survey.src_amp_x.param',
                'prop.model.survey.rec_loc_y',
                'prop.model.survey.rec_loc_x',
                'prop.model.survey.src_loc_y',
                'prop.model.survey.src_loc_x'
            ]
        )
    )
    return fwi_solver

def marmousi_acoustic_alan_check():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    u_obs = retrieve_dataset(field='u_obs', folder='marmousi', path='conda')
    vp = retrieve_dataset(field='vp', folder='marmousi', path='conda')
    
    ny = 600
    nx = 250
    nt = 300
    
    freq = 25
    peak_time = 1.5 / freq
    dt = 0.004

    rec_per_shot = 100
    n_shots = 20
    
    vp = vp[:ny, :nx]
    u_obs = u_obs[:n_shots, :rec_per_shot, :nt]

    survey = SurveyUniformLambda(
        n_shots=n_shots,
        src_y={
            'src_per_shot': 1,
            'fst_src': 10,
            'src_depth': 2,
            'd_src': 20,
            'd_intra_shot': 0
        },
        rec_y={
            'rec_per_shot': rec_per_shot,
            'fst_rec': 0,
            'rec_depth': 2,
            'd_rec': 6,
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
        y_amp_param=Param,
        x_amp_param=Param,
        ricker_freq=freq,
        peak_time=peak_time,
        nt=nt,
        dt=dt
    ).to(device)

    source_amplitudes = (
        (deepwave.wavelets.ricker(freq, nt, dt, peak_time))
        .repeat(n_shots, 1, 1)
    )

    n_sources_per_shot = 1
    d_source = 20
    first_source = 10
    source_depth = 2

    receiver_depth = 2
    d_receiver = 6
    first_receiver = 0
    n_receivers_per_shot = 100

    source_locations = torch.zeros(n_shots, n_sources_per_shot, 2,
                               dtype=torch.long, device=device)
    source_locations[..., 1] = source_depth
    source_locations[:, 0, 0] = (torch.arange(n_shots) * d_source +
                                first_source)

    # receiver_locations
    receiver_locations = torch.zeros(n_shots, n_receivers_per_shot, 2,
                                    dtype=torch.long, device=device)
    receiver_locations[..., 1] = receiver_depth
    receiver_locations[:, :, 0] = (
        (torch.arange(n_receivers_per_shot) * d_receiver +
        first_receiver)
        .repeat(n_shots, 1)
    ) 

    v_init = torch.tensor(1/gaussian_filter(1/vp.numpy(), 40))
    model = Model(
        survey=survey,
        model='acoustic',
        vp=v_init,
        rho=None,
        vs=None,
        vp_param=Param,
        vs_param=Param,
        rho_param=Param,
        freq=freq,
        dy=4.0,
        dx=4.0
    ).to(device)

    prop = Prop(
        model=model,
        train={
            'vp': True,
            'rho': False,
            'vs': False,
            'src_amp_y': False,
            'src_amp_x': False
        }
    )
    # deployer = Deployer(prop=prop, devices='all')
    # deployer = DeployerGPU(prop=prop, devices='all')
    # deployer = torch.nn.DataParallel(prop).to(device)
    # deployer = DeployerCPU(prop=prop)
    # deployer = DeployerIdentity(prop=prop, devices='ignore')

    inputd = lambda x, **kw: input(x)
    fwi = FWI(
        prop=prop,
        obs_data=u_obs.to(device),
        loss=torch.nn.MSELoss(),
        optimizer=[torch.optim.SGD, {'lr': 1e9, 'momentum': 0.9}],
        scheduler=[
            (torch.optim.lr_scheduler.ConstantLR, {'factor': 1.0})
        ],
        epochs=250,
        batch_size=10,
        verbosity='progress',
        print_protocol=print,
        make_plots=[('vp', True)],
        clip_grad=[('vp', 0.95)]
    )
    
    prop2 = fwi.prop
    debug = True
    if( debug ):
        print(torch.all(prop2.model.vp.param == v_init.to(device)))
        print(torch.all(prop2.model.survey.src_amp_y() == source_amplitudes.to(device)))
        print(torch.all(prop2.model.survey.src_loc_y == source_locations.to(device)))
        print(torch.all(prop2.model.survey.rec_loc_y == receiver_locations.to(device)))
        
        alan_optimizer = torch.optim.SGD([v_init], lr=1e9, momentum=0.9)
        alan_loss_fn = torch.nn.MSELoss()
        print(fwi.optimizer.param_groups)
        print(alan_optimizer.param_groups)
        print([e == f for (e,f) in zip(fwi.optimizer.param_groups, alan_optimizer.param_groups)])

        print(prop2.model.vp.param.requires_grad)
        print(prop2.model.vs.param.requires_grad)
        print(prop2.model.rho.param.requires_grad)
        print(prop2.model.survey.src_amp_y.param.requires_grad)
        print(prop2.model.survey.src_amp_x.param.requires_grad)
        print(prop2.model)

    return fwi

def marmousi_acoustic():
    return marmousi_acoustic_alan_check()