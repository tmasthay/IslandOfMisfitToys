"""
This script is the main workhorse of the Island of Misfit Toys.
It performs a complete full-waveform inversion workflow on any given dataset.
Arbitrary functions and keyword arguments can be passed into this script through Hydra's configuration hierarchy and command-line arguments.

Note:
    See `main_worker.py` for the main function that is called by this script.
    This dummy script is simply here to make autocompletion faster by bypassing import statements.

Usage:
    To see the full Hydra help menu, run the following command in terminal::

        python main.py --help

    For more information on Hydra, visit the official documentation:
    `Hydra Documentation <https://hydra.cc/docs/intro/>`_

Examples:
    Example usage of the script::

        python main.py case=frac case/data=full_marmousi case/train=max_iters=25

    This will run the script with the specified configuration groups and parameters.

    Example configuration groups::

        == Configuration groups ==
        Compose your configuration from those groups (group=option)

        case: frac, globals, l1, tik, w1, w2
        case/data: data, full_marmousi, shots16, two_layer
        case/data/preprocess: constrained_vel, w2
        case/data/preprocess/addons: softplus
        case/plt: plt
        case/train: train
        case/train/loss: frac, l1, l1_plain, relu, softplus, tik, transform_loss, w1
        case/train/optimizer: adam, gen, lbfgs
        case/train/stages: vanilla
        case/train/step: direct, relu, softplus, tik

    Example configuration override (long config example below)::

        == Config ==
        Override anything in the config (foo.bar=value)

        case:
        | port: 12576
        | dupe: true
        | editor: code
        | np: self.runtime.prop.module.meta.nt
        | data:
        | | path: conda/data/marmousi/deepwave_example/shots16
        | | preprocess:
        | | | dep: ^^null|misfit_toys.fwi.seismic_data|null
        | | | minv: 1000
        | | | maxv: 2500
        | | | time_pad_frac: 0.2
        | | | path_builder_kw:
        | | | | remap:
        | | | | | vp_init: vp
        | | | | vp_init:
        | | | | | runtime_func: self.data.preprocess.dep.ParamConstrained.delay_init
        | | | | | kw:
        | | | | | | minv: ${case.data.preprocess.minv}
        | | | | | | maxv: ${case.data.preprocess.maxv}
        | | | | | | requires_grad: true
        | | | | src_amp_y:
        | | | | | runtime_func: self.data.preprocess.dep.Param.delay_init
        | | | | | kw:
        | | | | | | requires_grad: false
        | | | | obs_data: null
        | | | | src_loc_y: null
        | | | | rec_loc_y: null
        | | | required_fields:
        | | | - vp_init
        | | | - src_amp_y
        | | | - obs_data
        | | | - src_loc_y
        | | | - rec_loc_y
        | | | - meta
        | | | chunk_keys:
        | | | | tensors:
        | | | | - obs_data
        | | | | - src_loc_y
        | | | | - rec_loc_y
        | | | | params:
        | | | | - src_amp_y
        | plt:
        | | vp:
        | | | sub:
        | | | | shape:
        | | | | - 2
        | | | | - 2
        | | | | kw:
        | | | | | figsize:
        | | | | | - 10
        | | | | | - 10
        | | | | adjust:
        | | | | | hspace: 0.5
        | | | | | wspace: 0.5
        | | | iter:
        | | | | none_dims:
        | | | | - -2
        | | | | - -1
        | | | save:
        | | | | path: figs/vp.gif
        | | | | movie_format: gif
        | | | | duration: 250
        | | | order:
        | | | - vp
        | | | - vp_true
        | | | - rel_diff
        | | | plts:
        | | | | vp:
        | | | | | main:
        | | | | | | filt: 'eval(lambda x : x.T)'
        | | | | | | opts:
        | | | | | | | cmap: seismic
        | | | | | | | aspect: auto
        | | | | | | title: $v_p$
        | | | | | | type: imshow
        | | | | | | xlabel: Rec Location (m)
        | | | | | | ylabel: Depth (m)
        | | | | | | colorbar: true
        | | | | rel_diff:
        | | | | | main:
        | | | | | | filt: transpose
        | | | | | | opts:
        | | | | | | | cmap: seismic
        | | | | | | | aspect: auto
        | | | | | | title: Relative Difference (%)
        | | | | | | type: imshow
        | | | | | | xlabel: Rec Location (m)
        | | | | | | ylabel: Depth (m)
        | | | | | | colorbar: true
        | | | | vp_true:
        | | | | | main:
        | | | | | | filt: transpose
        | | | | | | opts:
        | | | | | | | cmap: seismic
        | | | | | | | aspect: auto
        | | | | | | title: $v_{true}$
        | | | | | | type: imshow
        | | | | | | xlabel: Rec Location (m)
        | | | | | | ylabel: Depth (m)
        | | | | | | colorbar: true
        | | trace:
        | | | sub:
        | | | | shape:
        | | | | - 2
        | | | | - 2
        | | | | kw:
        | | | | | figsize:
        | | | | | - 10
        | | | | | - 10
        | | | iter:
        | | | | none_dims:
        | | | | - 0
        | | | | - -1
        | | | save:
        | | | | path: figs/random_traces.gif
        | | | | duration: 250
        | | | xlabel: Time (s)
        | | | ylabel: Displacement (m)
        | | | title: Observed Data at Receiver Location
        | | | color_seq:
        | | | - red
        | | | - blue
        | | | linestyles:
        | | | - solid
        | | | - dashed
        | | | legend:
        | | | | loc: upper right
        | | | | framealpha: 0.5
        | | | suptitle: Observed Data at Random Receiver Locations
        | train:
        | | retrain: true
        | | max_iters: 25
        | | loss:
        | | | dep:
        | | | | mod: ^^misfit_toys.fwi.loss.tikhonov
        | | | runtime_func: self.train.loss.dep.mod.TikhonovLoss
        | | | kw:
        | | | | runtime_func: self.train.loss.dep.mod.lin_reg_drop
        | | | | kw:
        | | | | | weights: self.runtime.prop.module.vp
        | | | | | max_iters: ${case.train.max_iters}
        | | | | | scale: 1.0e-06
        | | | | | _min: 1.0e-07
        | | optimizer:
        | | | runtime_func: 'eval(lambda *args, **kw: [torch.optim.LBFGS, kw])'
        | | | args: []
        | | | kw:
        | | | | lr: 1.0
        | | | | max_iter: 20
        | | | | max_eval: null
        | | | | tolerance_grad: 1.0e-07
        | | | | tolerance_change: 1.0e-09
        | | | | history_size: 100
        | | | | line_search_fn: null
        | | stages:
        | | | runtime_func: ^^null|misfit_toys.workflows.stages|vanilla_stages
        | | | kw:
        | | | | max_iters: ${case.train.max_iters}
        | | step:
        | | | runtime_func: ^^null|misfit_toys.workflows.tik.steps|taper_only
        | | | kw:
        | | | | length: 100
        | | | | num_batches: null
        | | | | scale: 1000000.0
        | | | nonexist: checking_the_yaml_format_hook
"""

import hydra
from omegaconf import DictConfig


@hydra.main(config_path="cfg", config_name="cfg", version_base=None)
def main_dummy(cfg: DictConfig) -> None:
    from main_worker import main

    main(cfg)


if __name__ == "__main__":
    main_dummy()
