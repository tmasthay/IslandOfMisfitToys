from elastic_custom import *
import matplotlib.pyplot as plt
from helpers.typlotlib import setup_gg_plot, rand_color, set_color_plot_global
import torch
import os
import deepwave

setup_gg_plot(clr_out=rand_color(), clr_in=rand_color())
config_plot = set_color_plot_global(use_legend=False)

def marmousi():
    def load_field(name):
        return torch.load(
            os.path.join(
                '/'.join(os.getcwd().split('/')[:-1]),
                'data/marmousi/torch_conversions/%s_marmousi2-ii.pt'%name
            )
        )
    vp = load_field('vp')
    vs = load_field('vs')
    rho = load_field('density')

    nt = 1000
    dt = 0.001
    dx = 1.25 #meters
    dy = 1.25 #meters

    n_shots = 1

    fst_src = vp.shape[1] // 2
    src_depth = 2
    d_src = 1
    n_src_per_shot = 1
    src_loc = torch.tensor(
        [
            [
                [src_depth, fst_src + d_src * i] \
                    for i in range(n_src_per_shot)
            ]
        ],
        device=device
    )

    fst_rec = 1
    rec_depth = 2
    d_rec = 1
    n_rec_per_shot = vp.shape[1]-2
    rec_loc = torch.tensor(
        [
            [
                [rec_depth, fst_rec + d_rec * i] \
                    for i in range(n_rec_per_shot)
            ]
        ],
        device=device
    )

    freq = 1.0 #Hz
    peak_time = 0.1 #seconds
    wavelet = deepwave.wavelets.ricker(freq, nt, dt, peak_time)

    ofs = 2

    class Marmousi(DataGenerator):
        def force(self, y, x, comp, *, my_other_kwarg=1):
            pass

        def forward(self):
            lamb_mu_buoy = deepwave.common.vpvsrho_to_lambmubuoyancy(
                self.vp,
                self.vs,
                self.rho
            )
            for (i,e) in enumerate(lamb_mu_buoy):
                print(f'{i}-->min={e.min()}, max={e.max()}')
            src_amp_y = 1e20 * wavelet \
                .unsqueeze(0) \
                .unsqueeze(0) \
                .expand(n_shots, n_src_per_shot, -1) \
                .to(device)
            return elastic(
                *lamb_mu_buoy,
                self.dx,
                self.dt,
                source_amplitudes_y=src_amp_y,
                source_locations_y=self.src_loc,
                receiver_locations_y=self.rec_loc,
                pml_freq=self.freq
            )[-2]

    return Marmousi(vp=vp,
        vs=vs,
        rho=rho,
        n_shots=n_shots,
        fst_src=fst_src,
        n_src_per_shot=n_src_per_shot,
        src_depth=src_depth,
        d_src=d_src,
        src_loc=src_loc,
        fst_rec=fst_rec,
        n_rec_per_shot=n_rec_per_shot,
        rec_depth=rec_depth,
        d_rec=d_rec,
        rec_loc=rec_loc,
        nt=nt,
        dt=dt,
        dx=dx,
        dy=dy,
        freq=freq,
        peak_time=peak_time,
        wavelet=wavelet,
        ofs=ofs
    )

def marmousi_dense_center_src():
    def load_field(name):
        return torch.load(
            os.path.join(
                '/'.join(os.getcwd().split('/')[:-1]),
                'data/marmousi/torch_conversions/%s_marmousi-ii.pt'%name
            )
        )
    vp = load_field('vp')
    vs = load_field('vs')
    rho = load_field('density')

    im1 = plt.imshow(vp.cpu(), cmap='jet', aspect='auto')
    config_plot(r'$V_p$')
    cbar1 = plt.colorbar(im1)
    cbar1.ax.tick_params(color='white')
    cbar1.ax.yaxis.set_tick_params(color='red')  
    plt.savefig('vp.png')
    plt.clf()

    im2 = plt.imshow(vs.cpu(), cmap='jet', aspect='auto')
    config_plot(r'$V_s$')
    cbar2 = plt.colorbar(im2)
    cbar2.ax.tick_params(color='white')
    cbar2.ax.yaxis.set_tick_params(color='red')
    plt.savefig('vs.png')
    plt.clf()

    im3 = plt.imshow(rho.cpu(), cmap='jet', aspect='auto')
    config_plot(r'$\rho$')
    cbar3 = plt.colorbar(im3)
    cbar3.ax.tick_params(color='white')
    cbar3.ax.yaxis.set_tick_params(color='red')
    plt.savefig('rho.png')
    plt.clf()

    print('Terminating early for debugging purposes')

    nt = 1000
    dt = 0.001
    dx = 1.25 #meters
    dy = 1.25 #meters

    ofs = 2
    ny_art = vp.shape[0] - ofs
    nx_art = vp.shape[1] - ofs

    # n_shots = ny_art
    n_shots = 1

    fst_src = vp.shape[1] // 2
    src_depth = vp.shape[0] // 2
    d_src = 1
    n_src_per_shot = 1
    src_loc = torch.tensor(
        [
            [
                [src_depth, fst_src + d_src * i] \
                    for i in range(n_src_per_shot)
            ]
            for _ in range(n_shots)
        ],
        device=device
    )
    fst_rec = 1
    fst_depth = 2
    rec_depth = fst_depth
    d_rec = 1
    d_depth = 1
    downsample_y = 10
    downsample_x = 10
    samples_y = ny_art // downsample_y
    samples_x = nx_art // downsample_x
    n_rec_per_shot = samples_y * samples_x
    # rec_loc = torch.tensor(
    #     [
    #         [
    #             [
    #                 fst_depth + d_depth * j, 
    #                 fst_rec + d_rec * i
    #             ] 
    #             for i in range(nx_art)
    #         ]
    #         for j in range(n_shots)
    #     ],
    #     device=device
    # )
    rec_loc = torch.tensor(
        [
            [
                [
                    fst_depth + d_depth * i,
                    fst_rec + d_rec * j
                ] 
                for i in range(samples_y)
            ]
            for j in range(samples_x)
        ],
        device=device
    ).reshape(1,samples_x*samples_y,2)

    freq = 10.0 #Hz
    peak_time = 0.1 #seconds
    wavelet = deepwave.wavelets.ricker(freq, nt, dt, peak_time)

    class Marmousi(DataGenerator):
        def __init__(self, **kw):
            super().__init__(**kw)

        def get(self, key):
            return self.custom[key]
        
        def force(self, y, x, comp, *, my_other_kwarg=1):
            pass

        def forward(self):
            lamb_mu_buoy = deepwave.common.vpvsrho_to_lambmubuoyancy(
                self.vp,
                self.vs,
                self.rho
            )
            for (i,e) in enumerate(lamb_mu_buoy):
                print(f'{i}-->min={e.min()}, max={e.max()}')
            src_amp_y = 1e20 * wavelet \
                .unsqueeze(0) \
                .unsqueeze(0) \
                .expand(n_shots, n_src_per_shot, -1) \
                .to(device)
            return elastic(
                *lamb_mu_buoy,
                self.dx,
                self.dt,
                source_amplitudes_y=src_amp_y,
                source_locations_y=self.src_loc,
                receiver_locations_y=self.rec_loc,
                pml_freq=self.freq
            )[-2]
        
        def update_depth(self, depth):
            self.rec_loc[:,:,0] = depth

    return Marmousi(vp=vp,
        vs=vs,
        rho=rho,
        n_shots=n_shots,
        fst_src=fst_src,
        n_src_per_shot=n_src_per_shot,
        src_depth=src_depth,
        d_src=d_src,
        src_loc=src_loc,
        fst_rec=fst_rec,
        n_rec_per_shot=n_rec_per_shot,
        rec_depth=rec_depth,
        d_rec=d_rec,
        rec_loc=rec_loc,
        nt=nt,
        dt=dt,
        dx=dx,
        dy=dy,
        freq=freq,
        peak_time=peak_time,
        wavelet=wavelet,
        ofs=ofs,
        samples_y=samples_y,
        samples_x=samples_x,
        downsample_x=downsample_x,
        downsample_y=downsample_y
    )

if( __name__ == '__main__' ):
    data = marmousi_dense_center_src()
    if( not os.path.exists('u.pt') ):
        print('building')
        # print('running')
        # delta = 1000
        # n_samples = (data.ny-data.ofs) // delta
        # input(n_samples)
        # input(data.n_shots)
        # input(data.n_rec_per_shot)
        # input(data.nt)

        # v = torch.empty(n_samples, data.n_shots, data.n_rec_per_shot, data.nt)
        # for i in range(n_samples):
        #     print(f'running {i}')
        #     data.update_depth(i + data.rec_depth * delta)
        #     v[i] = data.forward()
        #     plt.imshow(v[i,0], cmap='jet', aspect='auto')
        #     plt.savefig('u_%d.png'%(i*delta))
        # torch.save(v.cpu(), 'v.pt')
        print('running')
        u = data.forward()
        print(u.shape)
        torch.save(u.cpu(), 'u.pt')
    else:
        u = torch.load('u.pt')
        extent = [0, data.ny*data.dy, data.nx*data.dx, 0]
        for i in range(u.shape[0]):
            curr = u[i].reshape(
                data.get('samples_x'), 
                data.get('samples_y'), 
                data.nt
            )
            print(f'shot={i}')
            for j in range(data.nt):
                print(f'    time={j*data.dt}')
                plt.imshow(
                    curr[:,:,j], 
                    cmap='jet', 
                    aspect='auto', 
                    extent=extent
                )
                plt.title(f'time={j*data.dt}')
                plt.savefig('u_%d_%d.png'%(i,j))
                plt.clf()

        os.system('convert -delay 10 -loop 0 $(ls -tr u_*.png) u.gif')
