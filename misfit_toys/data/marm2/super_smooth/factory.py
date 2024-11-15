import os
from warnings import warn

import deepwave as dw
import torch
from deepwave.common import vpvsrho_to_lambmubuoyancy as get_lame

# add_root_package_path(path=os.path.dirname(__file__), pkg="misfit_toys")
from mh.core import DotDict

from misfit_toys.data.dataset import DataFactory, fixed_rec, towed_src
from misfit_toys.utils import select_best_gpu


def dsamp(tensor, *factors):
    assert all(
        isinstance(factor, int) for factor in factors
    ), 'Expected integer factors'
    assert len(factors) == tensor.ndim, 'Expected one factor per dimension'
    slices = tuple(slice(None, None, factor) for factor in factors)
    return tensor[slices]


class Factory(DataFactory):
    def _manufacture_data(self):
        if self.installed(
            "vp_true",
            "rho_true",
            "vs_true",
            "obs_data",
            "src_loc_y",
            "rec_loc_y",
            "src_amp_y",
            "src_loc_x",
            "src_amp_x",
            "rec_loc_x",
        ):
            return

        self.tensors = self.get_parent_tensors()
        d = DotDict(self.metadata)

        # _, src_slice, rec_slice = DataFactory.get_slices(d)
        # # self.slice_subset_tensors(
        # #     *v_slice, keys=["vp_true", "vs_true", "rho_true"]
        # # )
        # self.slice_subset_tensors(
        #     *src_slice,
        #     keys=["src_loc_y", "src_amp_y", "src_loc_x", "src_amp_x"],
        # )
        # self.slice_subset_tensors(*rec_slice, keys=["rec_loc_y", "rec_loc_x"])

        # domain downsamplers
        d.setdefault("down_y", 1)
        d.setdefault("down_x", 1)
        d.setdefault("down_t", 1)

        # downsample the acquisition geometry
        d.setdefault("down_shots", 1)
        d.setdefault("down_sps", 1)
        d.setdefault("down_rps", 1)

        self.tensors.vp_true = dsamp(self.tensors.vp_true, d.down_y, d.down_x)
        self.tensors.vs_true = dsamp(self.tensors.vs_true, d.down_y, d.down_x)
        self.tensors.rho_true = dsamp(self.tensors.rho_true, d.down_y, d.down_x)

        self.tensors.src_loc_y = dsamp(
            self.tensors.src_loc_y, d.down_shots, d.down_sps, 1
        )
        self.tensors.src_loc_x = dsamp(
            self.tensors.src_loc_x, d.down_shots, d.down_sps, 1
        )

        self.tensors.rec_loc_y = dsamp(
            self.tensors.rec_loc_y, d.down_shots, d.down_rps, 1
        )
        self.tensors.rec_loc_x = dsamp(
            self.tensors.rec_loc_x, d.down_shots, d.down_rps, 1
        )

        self.tensors.src_amp_y = dsamp(
            self.tensors.src_amp_y, d.down_shots, d.down_sps, 1
        )
        self.tensors.src_amp_x = dsamp(
            self.tensors.src_amp_x, d.down_shots, d.down_sps, 1
        )
        
        def contract_indices(x, *, clamp_y=None, clamp_x=None):
            x[..., 0] = x[..., 0] // d.down_y
            x[..., 1] = x[..., 1] // d.down_x
            
            if clamp_y is not None:
                x[..., 0] = torch.clamp(x[..., 0], min=clamp_y)
            if clamp_x is not None:
                x[..., 1] = torch.clamp(x[..., 1], min=clamp_x)
            return x

        self.tensors.src_loc_y = contract_indices(self.tensors.src_loc_y, clamp_y=1, clamp_x=1)
        self.tensors.src_loc_x = contract_indices(self.tensors.src_loc_x, clamp_y=1, clamp_x=1)
        
        self.tensors.rec_loc_y = contract_indices(self.tensors.rec_loc_y)
        self.tensors.rec_loc_x = contract_indices(self.tensors.rec_loc_x)
        
        delta = d.get('delta', 10)
        def trim_edges(x):
            N = x.shape[1]
            beta = min(delta, N // 2)
            if N <= 2:
                return x
            return x[:, beta:-beta, :]
        self.tensors.rec_loc_y = trim_edges(self.tensors.rec_loc_y)
        self.tensors.rec_loc_x = trim_edges(self.tensors.rec_loc_x)
        self.tensors.src_loc_y = trim_edges(self.tensors.src_loc_y)
        self.tensors.src_loc_x = trim_edges(self.tensors.src_loc_x)
        self.tensors.src_amp_y = trim_edges(self.tensors.src_amp_y)
        self.tensors.src_amp_x = trim_edges(self.tensors.src_amp_x)
        
        # input(f'{self.tensors.rec_loc_x.min()=}')
        # for k, v in self.tensors.items():
        #     input(f'{k=}, {v.shape=}')

        d.dy = d.dy * d.down_y
        d.dx = d.dx * d.down_x
        d.dt = d.dt * d.down_t

        print("Building obs_data in marmousi2/super_smooth...", end="", flush=True)
        res = dw.elastic(
            *get_lame(
                self.tensors.vp_true,
                self.tensors.vs_true,
                self.tensors.rho_true,
            ),
            d.dx,
            d.dt,
            source_amplitudes_y=self.tensors.src_amp_y,
            source_locations_y=self.tensors.src_loc_y,
            receiver_locations_y=self.tensors.rec_loc_y,
            source_amplitudes_x=self.tensors.src_amp_x,
            source_locations_x=self.tensors.src_loc_x,
            receiver_locations_x=self.tensors.rec_loc_x,
            pml_freq=d.freq,
            accuracy=d.accuracy,
        )
        self.tensors.obs_data = torch.stack(res[-2:], dim=-1).to('cpu')
        print("SUCCESS", flush=True)


def main():
    f = Factory.cli_construct(
        device="cuda:0", src_path=os.path.dirname(__file__)
    )
    f.manufacture_data()


if __name__ == "__main__":
    main()
