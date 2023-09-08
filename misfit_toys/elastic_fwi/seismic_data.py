import matplotlib.pyplot as plt
import torch
import os
from masthay_helpers.typlotlib import setup_gg_plot, rand_color, set_color_plot_global

from .elastic_class import *
from scipy.ndimage import gaussian_filter
from deepwave import scalar


def marmousi_acoustic():
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
    v_init = torch.tensor(1/gaussian_filter(1/vp_true.numpy(), 40))

    n_shots = 115

    n_sources_per_shot = 1
    d_source = 20  # 20 * 4m = 80m
    first_source = 10  # 10 * 4m = 40m
    source_depth = 2  # 2 * 4m = 8m

    n_receivers_per_shot = 384
    d_receiver = 6  # 6 * 4m = 24m
    first_receiver = 0  # 0 * 4m = 0m
    receiver_depth = 2  # 2 * 4m = 8m

    freq = 25
    nt = 750
    dt = 0.004
    peak_time = 1.5 / freq

    def taper(x):
        return deepwave.common.cosine_taper_end(x, 100)

    # Select portion of data for inversion
    n_receivers_per_shot = 100
    nt = 300

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

    data_path = os.path.join(path, 'marmousi_data.pt')
    if os.path.exists(data_path):
        print('Loading Marmousi data directly...')
        observed_data = torch.load(data_path)
    else:
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
    observed_data = taper(observed_data[:n_shots, :n_receivers_per_shot, :nt])
    
    devices = get_all_devices()
    vp_true = vp_true[:ny, :nx]
    v_init = v_init[:ny, :nx]

    msg1 = print
    msg1('CHECKPOINT 1')
    msg1(torch.cuda.memory_summary())

    ricker_freq = 25.0
    uniform_survey = SurveyUniformLambda(
        n_shots=4,
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
        y_amp_param=Param,
        x_amp_param=Param,
        ricker_freq=ricker_freq,
        peak_time=peak_time,
        nt=nt,
        dt=dt
    )

    msg2 = print
    msg2('CHECKPOINT 2: SURVEY')
    msg2(torch.cuda.memory_summary())

    pml_freq = ricker_freq
    model = Model(
        survey=uniform_survey,
        model='acoustic',
        vp=vp_true,
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

    msg3 = print
    msg3('CHECKPOINT 3: MODEL')
    msg3(torch.cuda.memory_summary())

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
    msg4 = input
    msg4('CHECKPOINT 4: PROP')
    msg4(torch.cuda.memory_summary())
    msg4(mem_report(*torch.cuda.mem_get_info(), rep=['free', 'total']))
    msg4(torch.cuda.list_gpu_processes())

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
        num_batches=n_shots,
        multi_gpu=False,
    )
    
    return fwi_solver