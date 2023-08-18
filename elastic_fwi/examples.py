from elastic_class import *
import matplotlib.pyplot as plt
from helpers.typlotlib import setup_gg_plot, rand_color, set_color_plot_global
import torch
import os
import deepwave

setup_gg_plot(clr_out='black', clr_in='black')
config_plot = set_color_plot_global(use_legend=False, use_colorbar=True, use_grid=False)

def marmousi():
    def load_field(name):
        return torch.load(
            os.path.join(
                '/'.join(os.getcwd().split('/')[:-1]),
                'data/marmousi2/torch_conversions/%s_marmousi-ii.pt'%name
            )
        )
    m_per_km = 1000.0
    vp = m_per_km * load_field('vp')
    vs = m_per_km * load_field('vs')
    rho = m_per_km * load_field('density')

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
                'data/marmousi2/torch_conversions/%s_marmousi-ii.pt'%name
            )
        )
    m_per_km = 1000.0
    vp = m_per_km * load_field('vp')
    vs = m_per_km * load_field('vs')
    rho = m_per_km * load_field('density')

    vp = 5000.0 * torch.ones_like(vp)
    vs = 3000.0 * torch.ones_like(vs)
    rho = (rho.max() + rho.min()) / 2.0 * torch.ones_like(rho)

    im1 = plt.imshow(vp.cpu(), cmap='jet', aspect='auto')
    config_plot(r'$V_p$')
    plt.savefig('vp.jpg')
    plt.clf()

    im2 = plt.imshow(vs.cpu(), cmap='jet', aspect='auto')
    config_plot(r'$V_s$')
    plt.savefig('vs.jpg')
    plt.clf()

    im3 = plt.imshow(rho.cpu(), cmap='jet', aspect='auto')
    config_plot(r'$\rho$')
    plt.savefig('rho.jpg')
    plt.clf()

    nt = 100
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
    peak_time = 1.5 / freq #seconds
    wavelet = deepwave.wavelets.ricker(freq, nt, dt, peak_time)

    class Marmousi(DataGenerator):
        def __init__(self, **kw):
            super().__init__(**kw)

        def get(self, key):
            return self.custom[key]
        
        def force(self, y, x, comp, *, my_other_kwarg=1):
            pass

        def forward(self):
            lamb, mu, buoy = deepwave.common.vpvsrho_to_lambmubuoyancy(
                self.vp,
                self.vs,
                self.rho
            )
            plt.imshow(lamb.cpu(), cmap='jet', aspect='auto')
            config_plot(r'$\lambda$')
            plt.savefig('lambda.jpg')
            plt.clf()
 
            plt.imshow(mu.cpu(), cmap='jet', aspect='auto')
            config_plot(r'$\mu$')
            plt.savefig('mu.jpg')
            plt.clf()

            plt.imshow(buoy.cpu(), cmap='jet', aspect='auto')
            config_plot(r'Buoyancy')
            plt.savefig('buoy.jpg')
            plt.clf()

            src_amp_y = 1e20 * wavelet \
                .unsqueeze(0) \
                .unsqueeze(0) \
                .expand(n_shots, n_src_per_shot, -1) \
                .to(device)
            return elastic(
                lamb,
                mu,
                buoy,
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

def acoustic_homogeneous():
    devices = get_all_devices()
    c = 1500.0
    ny = 250
    nx = 250
    vp  = c * torch.ones((nx,ny))
    vs = None
    rho = None

    nt = 1000
    dt = 0.001
    dx = 1.0 #km
    dy = 1.0 #km

    ofs = 2
    ny_art = vp.shape[0] - ofs
    nx_art = vp.shape[1] - ofs

    # n_shots = ny_art
    n_shots = 1

    d_src = 1
    n_src_per_shot = 1
    src_depth = vp.shape[0] // 2
    fst_src = vp.shape[1] // 2
    src_loc = uni_src_rec(
        n_shots=n_shots,
        idx_vert=[src_depth],
        idx_horz=[fst_src]
    ).to(devices[0])
    
    d_rec = 1
    n_rec_per_shot = 100
    rec_loc = uni_src_rec(
        n_shots=n_shots,
        idx_vert=range(ofs, vp.shape[1]-ofs, d_rec),
        idx_horz=range(ofs, vp.shape[0]-ofs, d_rec)
    ).to(devices[0])

    freq = 10.0 #Hz
    peak_time = 0.1 #seconds

    class Marmousi(DataGenerator):
        def __init__(self, **kw):
            super().__init__(**kw)
            self.src_amplitudes = self.force(
                self.src_loc[0,:,:]
            ) \
            .unsqueeze(0) \
            .to(self.devices[0])
            
            self.custom['samples_y'] = self.vp.shape[0] - 2 * self.ofs
            self.custom['samples_x'] = self.vp.shape[1] - 2 * self.ofs
            self.custom['downsample_y'] = 1
            self.custom['downsample_x'] = 1

        def get(self, key):
            return self.custom[key]
        
        def force(
            self, 
            p: Ant[torch.Tensor, 'Evaluation points'], 
            comp: Ant[str, 'Elastic component']='y', 
            *, 
            amp: Ant[float, 'Source amplitude']=1.0, 
            mu: Ant[list, 'Center of Gaussian']=[0.0,0.0], 
            sig: Ant[list, 'Stddev of Gaussian']=[1.0, 1.0]
        ): 
            G = amp * torch.exp( 
                -(p[:,0] - mu[0]) ** 2 / sig[0]**2 
                -(p[:,1] - mu[1]) ** 2 / sig[1]**2
            )
            return G.unsqueeze(-1) * self.wavelet.unsqueeze(0)

        def forward(self):
            return deepwave.scalar(
                self.vp,
                self.dx,
                self.dt,
                source_amplitudes=self.src_amplitudes,
                source_locations=self.src_loc,
                receiver_locations=self.rec_loc,
                pml_freq=self.freq
            )[-1]

    return Marmousi(vp=vp,
        vs=vs,
        rho=rho,
        n_shots=n_shots,
        fst_src=fst_src,
        n_src_per_shot=n_src_per_shot,
        src_depth=src_depth,
        d_src=d_src,
        src_loc=src_loc,
        fst_rec=ofs,
        n_rec_per_shot=n_rec_per_shot,
        rec_depth=ofs,
        d_rec=d_rec,
        rec_loc=rec_loc,
        nt=nt,
        dt=dt,
        dx=dx,
        dy=dy,
        freq=freq,
        peak_time=peak_time,
        ofs=ofs
    )


if( __name__ == '__main__' ):
    print('building')
    data = acoustic_homogeneous()
    if( not os.path.exists('u.pt') ):
        print('running')
        u = data.forward()
        print('saving')
        torch.save(u.cpu(), 'u.pt')
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument('--sub', type=int, default=1)

        args = parser.parse_args()

        u = torch.load('u.pt')
        extent = [0, data.ny*data.dy, data.nx*data.dx, 0]
        for i in range(u.shape[0]):
            curr = u[i].reshape(
                data.get('samples_x'), 
                data.get('samples_y'), 
                data.nt
            )
            print(f'shot={i}')
            for j in range(0, data.nt, args.sub):
                print(f'    time={j*data.dt}')
                plt.imshow(
                    curr[:,:,j], 
                    cmap='jet', 
                    aspect='auto', 
                    extent=extent
                )
                config_plot(f'time={j*data.dt}')
                plt.savefig('u_%d_%d.jpg'%(i,j))
                plt.clf()

            os.system('convert -delay 10 -loop 0 $(ls -tr u_%d_*.jpg) u.gif'%i)
