from elastic_custom import *
import matplotlib.pyplot as plt

def marmousi():
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

def marmousi_dense():
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
    fst_depth = 2
    rec_depth = fst_depth
    d_rec = 1
    d_depth = 1
    ofs = 2
    ny_art = vp.shape[0] - ofs
    nx_art = vp.shape[1] - ofs
    n_rec_per_shot = ny_art * nx_art
    rec_loc = torch.tensor(
        [
            [
                [
                    fst_depth + d_depth * i, 
                    fst_rec + d_rec * j
                ] 
                for j in range(ny_art)
            ]
            for i in range(nx_art)
        ]
    ).reshape((n_shots,n_rec_per_shot,2)).to(device)

    input(rec_loc.shape)

    freq = 1.0 #Hz
    peak_time = 0.1 #seconds
    wavelet = deepwave.wavelets.ricker(freq, nt, dt, peak_time)

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

if( __name__ == '__main__' ):
    print('building')
    data = marmousi_dense()
    print('running')
    u = data.forward()
    print(u.shape)
