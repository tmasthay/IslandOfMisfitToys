import matplotlib.pyplot as plt
import torch
import os
import deepwave

from masthay_helpers.typlotlib import setup_gg_plot, rand_color, set_color_plot_global

from .elastic_class import *

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
    idx_vert = range(ofs, vp.shape[1]-ofs, d_rec)
    idx_horz = range(ofs, vp.shape[0]-ofs, d_rec)
    samples_y = len(idx_vert)
    samples_x = len
    rec_loc = uni_src_rec(
        n_shots=n_shots,
        idx_vert=idx_vert,
        idx_horz=idx_horz
    ).to(devices[0])

    freq = 10.0 #Hz
    peak_time = 0.5 #seconds

    sig = [0.1, 0.1]
    amp = 1e3

    class Marmousi(DataGenerator):
        def __init__(self, **kw):
            super().__init__(**kw)
            self.src_amplitudes = self.force(
                self.src_loc[0,:,:],
                amp=kw['amp'],
                mu=kw['mu'],
                sig=kw['sig']
            ) \
            .unsqueeze(0) \
            .to(self.devices[0])

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
                -(p[:,0] * self.dy - mu[0]) ** 2 / sig[0]**2 
                -(p[:,1] * self.dx - mu[1]) ** 2 / sig[1]**2
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
        ofs=ofs,
        amp=amp,
        mu=[src_depth*dy, fst_src*dx],
        sig=sig,
        samples_y=samples_y,
        samples_x=samples_x
    )

def marmousi_real():
    devices = get_all_devices()
    def load_field(name):
        return torch.load(
            os.path.join(
                '/'.join(os.getcwd().split('/')[:-1]),
                'data/marmousi2/torch_conversions/%s_marmousi-ii.pt'%name
            )
        )
    m_per_km = 1000.0
    vp = m_per_km * load_field('vp')
    # vs = m_per_km * load_field('vs')
    # rho = m_per_km * load_field('density')
    vs = None
    rho = None

    ny = vp.shape[0]
    nx = vp.shape[1]

    nt = 2000
    dt = 0.001
    dx = 1.25 #km
    dy = 1.25 #km

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
    # idx_vert = range(ofs, vp.shape[0]-ofs, d_rec)
    idx_vert = [ofs]
    idx_horz = range(ofs, vp.shape[1]-ofs, d_rec)
    samples_y = len(idx_vert)
    samples_x = len(idx_horz)
    n_rec_per_shot = samples_y * samples_x
    rec_loc = uni_src_rec(
        n_shots=n_shots,
        idx_vert=idx_vert,
        idx_horz=idx_horz
    ).to(devices[0])

    freq = 10.0 #Hz
    peak_time = 0.5 #seconds

    sig = [0.1, 0.1]
    amp = 1e3

    class Marmousi(DataGenerator):
        def __init__(self, **kw):
            super().__init__(**kw)
            self.src_amplitudes = self.force(
                self.src_loc[0,:,:],
                amp=kw['amp'],
                mu=kw['mu'],
                sig=kw['sig']
            ) \
            .unsqueeze(0) \
            .to(self.devices[0])

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
                -(p[:,0] * self.dy - mu[0]) ** 2 / sig[0]**2 
                -(p[:,1] * self.dx - mu[1]) ** 2 / sig[1]**2
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
        ofs=ofs,
        amp=amp,
        mu=[src_depth*dy, fst_src*dx],
        sig=sig,
        samples_y=samples_y,
        samples_x=samples_x
    )

