from ..dataset import DataFactoryMeta, towed_src, fixed_rec, prettify_dict
from ...utils import DotDict
import os
import torch
from warnings import warn
import deepwave as dw
from scipy.ndimage import gaussian_filter
import copy


class Factory(DataFactoryMeta):
    def generate_derived_data(self, *, data):
        warn(
            '\n    This function contains a CUDA memory leak!'
            '\n    Run this prior to your main script and then use '
            '\n    get_data function to pull in obs_data.'
            '\n    Doing so will make this small memory leak only occur once'
            ' and be cleaned up by the garbage collector and thus benign.'
        )
        d = DotDict(data)

        ###NOTE: THIS IS THE DIFFERENCE! HE'S DOING FILTERING AFTER DOWNSAMPLING###
        ###MAKE SURE TO RECOMPUTE V_INIT FOR EACH OF THE DOWNSAMPLED CASES!###
        v_init = torch.tensor(1 / gaussian_filter(1 / d.vp_true.numpy(), 40))
        vp = d.vp_true.to(self.device)

        d.ny, d.nx = vp.shape

        src_loc_y = towed_src(
            n_shots=d.n_shots,
            src_per_shot=d.src_per_shot,
            d_src=d.d_src,
            fst_src=d.fst_src,
            src_depth=d.src_depth,
            d_intra_shot=d.d_intra_shot,
        ).to(self.device)

        rec_loc_y = fixed_rec(
            n_shots=d.n_shots,
            rec_per_shot=d.rec_per_shot,
            d_rec=d.d_rec,
            fst_rec=d.fst_rec,
            rec_depth=d.rec_depth,
        ).to(self.device)

        # source_amplitudes
        src_amp_y = (
            dw.wavelets.ricker(d.freq, d.nt, d.dt, d.peak_time)
            .repeat(d.n_shots, d.src_per_shot, 1)
            .to(self.device)
        )

        print(f'Building obs_data in {self.path}...', end='', flush=True)
        out = dw.scalar(
            vp,
            d.dy,
            d.dt,
            source_amplitudes=src_amp_y,
            source_locations=src_loc_y,
            receiver_locations=rec_loc_y,
            pml_freq=d.freq,
            accuracy=d.accuracy,
        )[-1]
        print('SUCCESS', flush=True)

        self.store_vars(
            ('obs_data', out),
            ('src_amp_y', src_amp_y),
            ('src_loc_y', src_loc_y),
            ('rec_loc_y', rec_loc_y),
            ('vp_init', v_init),
            subpath='',
        )

        self.downsample_all_paths(
            src_amp_y=src_amp_y,
            src_loc_y=src_loc_y,
            rec_loc_y=rec_loc_y,
            v_init=v_init,
            vp=vp,
            out=out,
        )

        del src_amp_y, src_loc_y, rec_loc_y, vp, v_init
        del out
        torch.cuda.empty_cache()

    def store_vars(self, *args, subpath=None):
        if subpath is None:
            subpath = ''
        for arg in args:
            path = os.path.join(self.path, subpath, arg[0])
            print(f'Saving {arg[0]} to {path}...', end='')
            filename = f'{path.replace(".pt","")}.pt'
            torch.save(arg[1].to('cpu'), filename)
            print('SUCCESS')

    def downsample_path(
        self, *, d, subpath, src_amp_y, src_loc_y, rec_loc_y, v_init, vp, out
    ):
        os.makedirs(os.path.join(self.path, subpath), exist_ok=True)

        der_dict = d[subpath]
        meta_dict_path = os.path.join(self.path, subpath, 'metadata.pydict')
        with open(meta_dict_path, 'w') as f:
            f.write(prettify_dict(der_dict))

        d_der = DotDict(der_dict)
        src_amp_y_der = src_amp_y[
            : d_der.n_shots, : d_der.src_per_shot, : d_der.nt
        ]
        src_loc_der = src_loc_y[: d_der.n_shots, : d_der.src_per_shot, :]
        rec_loc_der = rec_loc_y[: d_der.n_shots, : d_der.rec_per_shot, :]
        v_init_der = v_init[: d_der.ny, : d_der.nx]
        vp_der = vp[: d_der.ny, : d_der.nx]
        out_der = out[: d_der.n_shots, : d_der.rec_per_shot, : d_der.nt]

        self.store_vars(
            ('obs_data', out_der),
            ('src_amp_y', src_amp_y_der),
            ('src_loc_y', src_loc_der),
            ('rec_loc_y', rec_loc_der),
            ('vp_init', v_init_der),
            ('vp_true', vp_der),
            subpath=subpath,
        )

        if 'derived' in der_dict.keys():
            og_path = copy.deepcopy(self.path)
            self.path = os.path.join(self.path, subpath)
            subder_dict = DataFactoryMeta.get_derived_meta(meta=der_dict)
            self.downsample_all_paths(
                der_dict=subder_dict,
                src_amp_y=src_amp_y_der,
                src_loc_y=src_loc_der,
                rec_loc_y=rec_loc_der,
                v_init=v_init_der,
                vp=vp_der,
                out=out_der,
            )
            self.path = og_path

    def downsample_all_paths(
        self, *, der_dict=None, src_amp_y, src_loc_y, rec_loc_y, v_init, vp, out
    ):
        if der_dict is None:
            der_dict = DataFactoryMeta.get_derived_meta(meta=self.metadata)
        for subpath in der_dict.keys():
            self.downsample_path(
                d=der_dict,
                subpath=subpath,
                src_amp_y=src_amp_y,
                src_loc_y=src_loc_y,
                rec_loc_y=rec_loc_y,
                v_init=v_init,
                vp=vp,
                out=out,
            )
