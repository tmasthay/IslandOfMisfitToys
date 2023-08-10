from elastic_custom import *

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

    n_shots = 1

    fst_src = vp.shape[1] // 2
    src_depth = 2
    d_src = 1
    n_src_per_shot = 1
    src_loc = torch.tensor([fst_src + d_src * i for i in range(n_src_per_shot)])

    fst_rec = vp.shape[1] // 2
    rec_depth = 2
    d_rec = 1
    n_rec_per_shot = 100
    rec_loc = torch.tensor([fst_rec + d_rec * i for i in range(n_rec_per_shot)])

    nt = 1000
    dt = 0.001
    dx = 25 #meters
    dy = 25 #meters

    freq = 10 #Hz
    peak_time = 0.1 #seconds
    wavelet = deepwave.wavelets.ricker(freq, nt, dt, peak_time)

    ofs = 2

    class Marmousi(DataGenerator):
        def force(self, y, x, comp, *, my_other_kwarg=1):
            pass

        def forward(self, *, my_kwarg=1):
            lamb_mu_buoy = deepwave.common.vpvsrho_to_lambmubuoyancy(
                self.vp,
                self.vs,
                self.rho
            )
            pass

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
    data = marmousi()
    print(data) 
