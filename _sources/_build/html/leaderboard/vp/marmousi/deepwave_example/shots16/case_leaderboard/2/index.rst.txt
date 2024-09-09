2
=

.. toctree::
   :maxdepth: 4
   :caption: Contents:


Metadata
--------


.. admonition:: Metadata
  :class: toggle


  .. admonition:: score
    :class: toggle


    .. admonition:: vp_compare.yaml
      :class: toggle


      .. code-block:: yaml

        human_timestamp: July 25, 2024 09:57:15 AM
        l2_diff: 48.273643493652344
        max_iters: 25
        name: tik
        orig_root: /home/tyler/Documents/repos/IslandOfMisfitToys/misfit_toys/hydra/outputs/HYDRA_TIME_2024-07-25/HYDRA_TIME_09-57-15/marmousi/deepwave_example/shots16/
        proj_path: marmousi/deepwave_example/shots16
        root: ../../misfit_toys/hydra/outputs/HYDRA_TIME_2024-07-25/HYDRA_TIME_09-57-15/marmousi/deepwave_example/shots16
        timestamp: 2024-07-25 09-57-15
        train_time: 110.87736320495605


  .. admonition:: hyperparameters
    :class: toggle


    .. admonition:: config.yaml
      :class: toggle


      .. code-block:: yaml

        case:
          port: 12576
          dupe: true
          editor: code
          name: tik
          np: self.runtime.data.meta.nt
          data:
            prefix: conda/data
            proj_path: marmousi/deepwave_example/shots16
            path: ${.prefix}/${.proj_path}
            preprocess:
              dep: ^^null|misfit_toys.fwi.seismic_data|null
              minv: 1000
              maxv: 2500
              time_pad_frac: 0.2
              path_builder_kw:
                remap:
                  vp_init: vp
                vp_init:
                  runtime_func: self.data.preprocess.dep.ParamConstrained.delay_init
                  kw:
                    minv: ${case.data.preprocess.minv}
                    maxv: ${case.data.preprocess.maxv}
                    requires_grad: true
                src_amp_y:
                  runtime_func: self.data.preprocess.dep.Param.delay_init
                  kw:
                    requires_grad: false
                obs_data: null
                src_loc_y: null
                rec_loc_y: null
              required_fields:
              - vp_init
              - src_amp_y
              - obs_data
              - src_loc_y
              - rec_loc_y
              - meta
              chunk_keys:
                tensors:
                - obs_data
                - src_loc_y
                - rec_loc_y
                params:
                - src_amp_y
            postprocess:
              __call__: ^^null|misfit_toys.beta.postprocess|vp_compare
              kw:
                proj_path: ${...proj_path}
                name: ${case.name}
                max_iters: ${case.train.max_iters}
          plt:
            vp:
              sub:
                shape:
                - 2
                - 2
                kw:
                  figsize:
                  - 10
                  - 10
                adjust:
                  hspace: 0.5
                  wspace: 0.5
              iter:
                none_dims:
                - -2
                - -1
              save:
                path: figs/vp.gif
                movie_format: gif
                duration: 250
              order:
              - vp
              - vp_true
              - rel_diff
              plts:
                vp:
                  main:
                    filt: 'eval(lambda x : x.T)'
                    opts:
                      cmap: seismic
                      aspect: auto
                    title: $v_p$
                    type: imshow
                    xlabel: Rec Location (m)
                    ylabel: Depth (m)
                    colorbar: true
                rel_diff:
                  main:
                    filt: transpose
                    opts:
                      cmap: seismic
                      aspect: auto
                    title: Relative Difference (%)
                    type: imshow
                    xlabel: Rec Location (m)
                    ylabel: Depth (m)
                    colorbar: true
                vp_true:
                  main:
                    filt: transpose
                    opts:
                      cmap: seismic
                      aspect: auto
                    title: $v_{true}$
                    type: imshow
                    xlabel: Rec Location (m)
                    ylabel: Depth (m)
                    colorbar: true
            trace:
              sub:
                shape:
                - 2
                - 2
                kw:
                  figsize:
                  - 10
                  - 10
              iter:
                none_dims:
                - 0
                - -1
              save:
                path: figs/random_traces.gif
                duration: 250
              xlabel: Time (s)
              ylabel: Displacement (m)
              title: Observed Data at Receiver Location
              color_seq:
              - red
              - blue
              linestyles:
              - solid
              - dashed
              legend:
                loc: upper right
                framealpha: 0.5
              suptitle: Observed Data at Random Receiver Locations
          train:
            retrain: true
            max_iters: 25
            loss:
              dep:
                mod: ^^misfit_toys.fwi.loss.tikhonov
              runtime_func: self.train.loss.dep.mod.TikhonovLoss
              kw:
                runtime_func: self.train.loss.dep.mod.lin_reg_drop
                kw:
                  weights: self.runtime.prop.module.vp
                  max_iters: ${case.train.max_iters}
                  scale: 1.0e-06
                  _min: 1.0e-07
            optimizer:
              runtime_func: 'eval(lambda *args, **kw: [torch.optim.LBFGS, kw])'
              args: []
              kw:
                lr: 1.0
                max_iter: 20
                max_eval: null
                tolerance_grad: 1.0e-07
                tolerance_change: 1.0e-09
                history_size: 100
                line_search_fn: null
            stages:
              runtime_func: ^^null|misfit_toys.workflows.stages|vanilla_stages
              kw:
                max_iters: ${case.train.max_iters}
            step:
              runtime_func: ^^null|misfit_toys.workflows.tik.steps|taper_only
              kw:
                length: 100
                num_batches: null
                scale: 1000000.0
                src_amp_y: self.runtime.data.src_amp_y
              nonexist: checking_the_yaml_format_hook
            prop:
              runtime_func: ^^|misfit_toys.fwi.seismic_data|DebugProp
              kw:
                vp: self.runtime.data.vp
                dx: self.runtime.data.meta.dx
                dt: self.runtime.data.meta.dt
                freq: self.runtime.data.meta.freq
                rec_loc_y: self.runtime.data.rec_loc_y
                src_loc_y: self.runtime.data.src_loc_y
        run: test


    .. admonition:: overrides.yaml
      :class: toggle


      .. code-block:: yaml

        - +run=test


  .. admonition:: version control
    :class: toggle


    .. admonition:: git_info.txt
      :class: toggle


      .. code-block:: text

        HASH: e8cf98d49e0be37cd9c76434651feccdae1d8dea
        BRANCH: feature/full_marmousi_debug_cleanup
        
        UNTRACKED FILES: .latest_run
        .vscode/settings.json
        
        ********************************************************************************
        DIFF: 
        ********************************************************************************


  .. admonition:: stdout
    :class: toggle


    .. admonition:: main.log
      :class: toggle


      .. code-block:: text

        Empty file


    .. admonition:: rank_0.out
      :class: toggle


      .. code-block:: text

        Preprocessing took 0.01 seconds.
           Loading /home/tyler/anaconda3/envs/dw/data/marmousi/deepwave_example/shots16/vp_init.pt...torch.Size([600, 250])
           Loading /home/tyler/anaconda3/envs/dw/data/marmousi/deepwave_example/shots16/src_amp_y.pt...torch.Size([16, 1, 300])
           Loading /home/tyler/anaconda3/envs/dw/data/marmousi/deepwave_example/shots16/obs_data.pt...torch.Size([16, 100, 300])
           Loading /home/tyler/anaconda3/envs/dw/data/marmousi/deepwave_example/shots16/src_loc_y.pt...torch.Size([16, 1, 2])
           Loading /home/tyler/anaconda3/envs/dw/data/marmousi/deepwave_example/shots16/rec_loc_y.pt...torch.Size([16, 100, 2])
        Preprocess time rank 0: 1.07 seconds.
        iter=0, loss=5.78e-03, mse=5.78e-03, tik=9.36e-07, reg_strength=1.00e-06, training.loss: 5.78e+03, lr: 1.000e+00, obs_data.norm: 4.05e+01, out.norm: 3.99e+01, rank: 0
        iter=1, loss=2.81e-03, mse=2.48e-03, tik=3.27e-04, reg_strength=9.60e-07, training.loss: 2.81e+03, lr: 1.000e+00, obs_data.norm: 4.05e+01, out.norm: 2.97e+01, rank: 0
        iter=2, loss=2.02e-03, mse=1.74e-03, tik=2.77e-04, reg_strength=9.20e-07, training.loss: 2.02e+03, lr: 1.000e+00, obs_data.norm: 4.05e+01, out.norm: 3.42e+01, rank: 0
        iter=3, loss=9.03e-04, mse=7.44e-04, tik=1.60e-04, reg_strength=8.80e-07, training.loss: 9.03e+02, lr: 1.000e+00, obs_data.norm: 4.05e+01, out.norm: 3.89e+01, rank: 0
        iter=4, loss=3.55e-04, mse=2.68e-04, tik=8.67e-05, reg_strength=8.40e-07, training.loss: 3.55e+02, lr: 1.000e+00, obs_data.norm: 4.05e+01, out.norm: 3.94e+01, rank: 0
        iter=5, loss=1.73e-04, mse=1.27e-04, tik=4.63e-05, reg_strength=8.00e-07, training.loss: 1.73e+02, lr: 1.000e+00, obs_data.norm: 4.05e+01, out.norm: 4.00e+01, rank: 0
        iter=6, loss=9.11e-05, mse=7.07e-05, tik=2.04e-05, reg_strength=7.60e-07, training.loss: 9.11e+01, lr: 1.000e+00, obs_data.norm: 4.05e+01, out.norm: 4.03e+01, rank: 0
        iter=7, loss=5.65e-05, mse=4.58e-05, tik=1.06e-05, reg_strength=7.20e-07, training.loss: 5.65e+01, lr: 1.000e+00, obs_data.norm: 4.05e+01, out.norm: 4.04e+01, rank: 0
        iter=8, loss=3.91e-05, mse=2.94e-05, tik=9.72e-06, reg_strength=6.80e-07, training.loss: 3.91e+01, lr: 1.000e+00, obs_data.norm: 4.05e+01, out.norm: 4.05e+01, rank: 0
        iter=9, loss=3.05e-05, mse=2.15e-05, tik=9.08e-06, reg_strength=6.40e-07, training.loss: 3.05e+01, lr: 1.000e+00, obs_data.norm: 4.05e+01, out.norm: 4.06e+01, rank: 0
        iter=10, loss=2.57e-05, mse=1.75e-05, tik=8.28e-06, reg_strength=6.00e-07, training.loss: 2.57e+01, lr: 1.000e+00, obs_data.norm: 4.05e+01, out.norm: 4.06e+01, rank: 0
        iter=11, loss=2.33e-05, mse=1.57e-05, tik=7.55e-06, reg_strength=5.60e-07, training.loss: 2.33e+01, lr: 1.000e+00, obs_data.norm: 4.05e+01, out.norm: 4.06e+01, rank: 0
        iter=12, loss=2.16e-05, mse=1.45e-05, tik=7.08e-06, reg_strength=5.20e-07, training.loss: 2.16e+01, lr: 1.000e+00, obs_data.norm: 4.05e+01, out.norm: 4.06e+01, rank: 0
        iter=13, loss=1.97e-05, mse=1.30e-05, tik=6.71e-06, reg_strength=4.80e-07, training.loss: 1.97e+01, lr: 1.000e+00, obs_data.norm: 4.05e+01, out.norm: 4.06e+01, rank: 0
        iter=14, loss=1.82e-05, mse=1.18e-05, tik=6.42e-06, reg_strength=4.40e-07, training.loss: 1.82e+01, lr: 1.000e+00, obs_data.norm: 4.05e+01, out.norm: 4.06e+01, rank: 0
        iter=15, loss=1.71e-05, mse=1.10e-05, tik=6.04e-06, reg_strength=4.00e-07, training.loss: 1.71e+01, lr: 1.000e+00, obs_data.norm: 4.05e+01, out.norm: 4.06e+01, rank: 0
        iter=16, loss=1.61e-05, mse=1.03e-05, tik=5.74e-06, reg_strength=3.60e-07, training.loss: 1.61e+01, lr: 1.000e+00, obs_data.norm: 4.05e+01, out.norm: 4.06e+01, rank: 0
        iter=17, loss=1.51e-05, mse=9.77e-06, tik=5.35e-06, reg_strength=3.20e-07, training.loss: 1.51e+01, lr: 1.000e+00, obs_data.norm: 4.05e+01, out.norm: 4.06e+01, rank: 0
        iter=18, loss=1.41e-05, mse=9.18e-06, tik=4.87e-06, reg_strength=2.80e-07, training.loss: 1.41e+01, lr: 1.000e+00, obs_data.norm: 4.05e+01, out.norm: 4.06e+01, rank: 0
        iter=19, loss=1.29e-05, mse=8.59e-06, tik=4.34e-06, reg_strength=2.40e-07, training.loss: 1.29e+01, lr: 1.000e+00, obs_data.norm: 4.05e+01, out.norm: 4.06e+01, rank: 0
        iter=20, loss=1.20e-05, mse=8.24e-06, tik=3.74e-06, reg_strength=2.00e-07, training.loss: 1.20e+01, lr: 1.000e+00, obs_data.norm: 4.05e+01, out.norm: 4.06e+01, rank: 0
        iter=21, loss=1.11e-05, mse=7.93e-06, tik=3.13e-06, reg_strength=1.60e-07, training.loss: 1.11e+01, lr: 1.000e+00, obs_data.norm: 4.05e+01, out.norm: 4.06e+01, rank: 0
        iter=22, loss=1.02e-05, mse=7.70e-06, tik=2.47e-06, reg_strength=1.20e-07, training.loss: 1.02e+01, lr: 1.000e+00, obs_data.norm: 4.05e+01, out.norm: 4.06e+01, rank: 0
        iter=23, loss=9.65e-06, mse=7.48e-06, tik=2.17e-06, reg_strength=1.00e-07, training.loss: 9.65e+00, lr: 1.000e+00, obs_data.norm: 4.05e+01, out.norm: 4.06e+01, rank: 0
        iter=24, loss=9.58e-06, mse=7.33e-06, tik=2.25e-06, reg_strength=1.00e-07, training.loss: 9.58e+00, lr: 1.000e+00, obs_data.norm: 4.05e+01, out.norm: 4.06e+01, rank: 0
        Presaving loss on rank 0
        Presaving vp on rank 0
        Presaving out on rank 0
        Presaving loss on rank 0
        Presaving vp on rank 0
        Presaving out on rank 0
        yo
        destroyed
        Train time rank 0: 106.20 seconds.


    .. admonition:: rank_1.out
      :class: toggle


      .. code-block:: text

        Preprocessing took 0.01 seconds.
           Loading /home/tyler/anaconda3/envs/dw/data/marmousi/deepwave_example/shots16/vp_init.pt...torch.Size([600, 250])
           Loading /home/tyler/anaconda3/envs/dw/data/marmousi/deepwave_example/shots16/src_amp_y.pt...torch.Size([16, 1, 300])
           Loading /home/tyler/anaconda3/envs/dw/data/marmousi/deepwave_example/shots16/obs_data.pt...torch.Size([16, 100, 300])
           Loading /home/tyler/anaconda3/envs/dw/data/marmousi/deepwave_example/shots16/src_loc_y.pt...torch.Size([16, 1, 2])
           Loading /home/tyler/anaconda3/envs/dw/data/marmousi/deepwave_example/shots16/rec_loc_y.pt...torch.Size([16, 100, 2])
        Preprocess time rank 1: 1.06 seconds.
        iter=0, loss=7.53e-03, mse=7.53e-03, tik=9.36e-07, reg_strength=1.00e-06, training.loss: 7.53e+03, lr: 1.000e+00, obs_data.norm: 4.18e+01, out.norm: 4.01e+01, rank: 1
        iter=1, loss=3.77e-03, mse=3.44e-03, tik=3.27e-04, reg_strength=9.60e-07, training.loss: 3.77e+03, lr: 1.000e+00, obs_data.norm: 4.18e+01, out.norm: 2.56e+01, rank: 1
        iter=2, loss=2.58e-03, mse=2.30e-03, tik=2.77e-04, reg_strength=9.20e-07, training.loss: 2.58e+03, lr: 1.000e+00, obs_data.norm: 4.18e+01, out.norm: 3.31e+01, rank: 1
        iter=3, loss=1.15e-03, mse=9.93e-04, tik=1.60e-04, reg_strength=8.80e-07, training.loss: 1.15e+03, lr: 1.000e+00, obs_data.norm: 4.18e+01, out.norm: 3.98e+01, rank: 1
        iter=4, loss=4.43e-04, mse=3.57e-04, tik=8.67e-05, reg_strength=8.40e-07, training.loss: 4.43e+02, lr: 1.000e+00, obs_data.norm: 4.18e+01, out.norm: 4.00e+01, rank: 1
        iter=5, loss=2.16e-04, mse=1.70e-04, tik=4.63e-05, reg_strength=8.00e-07, training.loss: 2.16e+02, lr: 1.000e+00, obs_data.norm: 4.18e+01, out.norm: 4.10e+01, rank: 1
        iter=6, loss=1.14e-04, mse=9.36e-05, tik=2.04e-05, reg_strength=7.60e-07, training.loss: 1.14e+02, lr: 1.000e+00, obs_data.norm: 4.18e+01, out.norm: 4.13e+01, rank: 1
        iter=7, loss=6.72e-05, mse=5.66e-05, tik=1.06e-05, reg_strength=7.20e-07, training.loss: 6.72e+01, lr: 1.000e+00, obs_data.norm: 4.18e+01, out.norm: 4.16e+01, rank: 1
        iter=8, loss=4.39e-05, mse=3.42e-05, tik=9.72e-06, reg_strength=6.80e-07, training.loss: 4.39e+01, lr: 1.000e+00, obs_data.norm: 4.18e+01, out.norm: 4.16e+01, rank: 1
        iter=9, loss=3.24e-05, mse=2.33e-05, tik=9.08e-06, reg_strength=6.40e-07, training.loss: 3.24e+01, lr: 1.000e+00, obs_data.norm: 4.18e+01, out.norm: 4.17e+01, rank: 1
        iter=10, loss=2.71e-05, mse=1.88e-05, tik=8.28e-06, reg_strength=6.00e-07, training.loss: 2.71e+01, lr: 1.000e+00, obs_data.norm: 4.18e+01, out.norm: 4.17e+01, rank: 1
        iter=11, loss=2.42e-05, mse=1.66e-05, tik=7.55e-06, reg_strength=5.60e-07, training.loss: 2.42e+01, lr: 1.000e+00, obs_data.norm: 4.18e+01, out.norm: 4.17e+01, rank: 1
        iter=12, loss=2.25e-05, mse=1.54e-05, tik=7.08e-06, reg_strength=5.20e-07, training.loss: 2.25e+01, lr: 1.000e+00, obs_data.norm: 4.18e+01, out.norm: 4.17e+01, rank: 1
        iter=13, loss=2.11e-05, mse=1.44e-05, tik=6.71e-06, reg_strength=4.80e-07, training.loss: 2.11e+01, lr: 1.000e+00, obs_data.norm: 4.18e+01, out.norm: 4.18e+01, rank: 1
        iter=14, loss=1.97e-05, mse=1.33e-05, tik=6.42e-06, reg_strength=4.40e-07, training.loss: 1.97e+01, lr: 1.000e+00, obs_data.norm: 4.18e+01, out.norm: 4.18e+01, rank: 1
        iter=15, loss=1.82e-05, mse=1.21e-05, tik=6.04e-06, reg_strength=4.00e-07, training.loss: 1.82e+01, lr: 1.000e+00, obs_data.norm: 4.18e+01, out.norm: 4.18e+01, rank: 1
        iter=16, loss=1.68e-05, mse=1.11e-05, tik=5.74e-06, reg_strength=3.60e-07, training.loss: 1.68e+01, lr: 1.000e+00, obs_data.norm: 4.18e+01, out.norm: 4.18e+01, rank: 1
        iter=17, loss=1.53e-05, mse=9.94e-06, tik=5.35e-06, reg_strength=3.20e-07, training.loss: 1.53e+01, lr: 1.000e+00, obs_data.norm: 4.18e+01, out.norm: 4.18e+01, rank: 1
        iter=18, loss=1.37e-05, mse=8.78e-06, tik=4.87e-06, reg_strength=2.80e-07, training.loss: 1.37e+01, lr: 1.000e+00, obs_data.norm: 4.18e+01, out.norm: 4.18e+01, rank: 1
        iter=19, loss=1.23e-05, mse=7.98e-06, tik=4.34e-06, reg_strength=2.40e-07, training.loss: 1.23e+01, lr: 1.000e+00, obs_data.norm: 4.18e+01, out.norm: 4.18e+01, rank: 1
        iter=20, loss=1.11e-05, mse=7.32e-06, tik=3.74e-06, reg_strength=2.00e-07, training.loss: 1.11e+01, lr: 1.000e+00, obs_data.norm: 4.18e+01, out.norm: 4.18e+01, rank: 1
        iter=21, loss=1.00e-05, mse=6.87e-06, tik=3.13e-06, reg_strength=1.60e-07, training.loss: 1.00e+01, lr: 1.000e+00, obs_data.norm: 4.18e+01, out.norm: 4.18e+01, rank: 1
        iter=22, loss=9.02e-06, mse=6.54e-06, tik=2.47e-06, reg_strength=1.20e-07, training.loss: 9.02e+00, lr: 1.000e+00, obs_data.norm: 4.18e+01, out.norm: 4.18e+01, rank: 1
        iter=23, loss=8.48e-06, mse=6.31e-06, tik=2.17e-06, reg_strength=1.00e-07, training.loss: 8.48e+00, lr: 1.000e+00, obs_data.norm: 4.18e+01, out.norm: 4.18e+01, rank: 1
        iter=24, loss=8.34e-06, mse=6.09e-06, tik=2.25e-06, reg_strength=1.00e-07, training.loss: 8.34e+00, lr: 1.000e+00, obs_data.norm: 4.18e+01, out.norm: 4.18e+01, rank: 1
        Presaving loss on rank 1
        Presaving vp on rank 1
        Presaving out on rank 1
        Presaving loss on rank 1
        Presaving vp on rank 1
        Presaving out on rank 1
        yo
        destroyed
        Train time rank 1: 106.20 seconds.


  .. admonition:: stderr
    :class: toggle


    .. admonition:: rank_0.err
      :class: toggle


      .. code-block:: text

        Empty file


    .. admonition:: rank_1.err
      :class: toggle


      .. code-block:: text

        Empty file


  .. admonition:: other
    :class: toggle


    .. admonition:: hydra.yaml
      :class: toggle


      .. code-block:: yaml

        hydra:
          run:
            dir: outputs/HYDRA_TIME_${now:%Y-%m-%d}/HYDRA_TIME_${now:%H-%M-%S}/${case.data.proj_path}
          sweep:
            dir: multirun/HYDRA_TIME_${now:%Y-%m-%d}/HYDRA_TIME_${now:%H-%M-%S}/${case.data.proj_path}
            subdir: ${hydra.job.num}
          launcher:
            _target_: hydra._internal.core_plugins.basic_launcher.BasicLauncher
          sweeper:
            _target_: hydra._internal.core_plugins.basic_sweeper.BasicSweeper
            max_batch_size: null
            params: null
          help:
            app_name: ${hydra.job.name}
            header: '${hydra.help.app_name} is powered by Hydra.
        
              '
            footer: 'Powered by Hydra (https://hydra.cc)
        
              Use --hydra-help to view Hydra specific help
        
              '
            template: '${hydra.help.header}
        
              == Configuration groups ==
        
              Compose your configuration from those groups (group=option)
        
        
              $APP_CONFIG_GROUPS
        
        
              == Config ==
        
              Override anything in the config (foo.bar=value)
        
        
              $CONFIG
        
        
              ${hydra.help.footer}
        
              '
          hydra_help:
            template: 'Hydra (${hydra.runtime.version})
        
              See https://hydra.cc for more info.
        
        
              == Flags ==
        
              $FLAGS_HELP
        
        
              == Configuration groups ==
        
              Compose your configuration from those groups (For example, append hydra/job_logging=disabled
              to command line)
        
        
              $HYDRA_CONFIG_GROUPS
        
        
              Use ''--cfg hydra'' to Show the Hydra config.
        
              '
            hydra_help: ???
          hydra_logging:
            version: 1
            formatters:
              simple:
                format: '[%(asctime)s][HYDRA] %(message)s'
            handlers:
              console:
                class: logging.StreamHandler
                formatter: simple
                stream: ext://sys.stdout
            root:
              level: INFO
              handlers:
              - console
            loggers:
              logging_example:
                level: DEBUG
            disable_existing_loggers: false
          job_logging:
            version: 1
            formatters:
              simple:
                format: '[%(asctime)s][%(name)s][%(levelname)s] - %(message)s'
            handlers:
              console:
                class: logging.StreamHandler
                formatter: simple
                stream: ext://sys.stdout
              file:
                class: logging.FileHandler
                formatter: simple
                filename: ${hydra.runtime.output_dir}/${hydra.job.name}.log
            root:
              level: INFO
              handlers:
              - console
              - file
            disable_existing_loggers: false
          env: {}
          mode: RUN
          searchpath: []
          callbacks: {}
          output_subdir: ''
          overrides:
            hydra:
            - hydra.mode=RUN
            task:
            - +run=test
          job:
            name: main
            chdir: null
            override_dirname: +run=test
            id: ???
            num: ???
            config_name: cfg
            env_set: {}
            env_copy: []
            config:
              override_dirname:
                kv_sep: '='
                item_sep: ','
                exclude_keys: []
          runtime:
            version: 1.3.2
            version_base: '1.3'
            cwd: /home/tyler/Documents/repos/IslandOfMisfitToys/misfit_toys/hydra
            config_sources:
            - path: hydra.conf
              schema: pkg
              provider: hydra
            - path: /home/tyler/Documents/repos/IslandOfMisfitToys/misfit_toys/hydra/cfg
              schema: file
              provider: main
            - path: ''
              schema: structured
              provider: schema
            output_dir: /home/tyler/Documents/repos/IslandOfMisfitToys/misfit_toys/hydra/outputs/HYDRA_TIME_2024-07-25/HYDRA_TIME_09-57-15/marmousi/deepwave_example/shots16
            choices:
              case: tik
              case/train/prop: refactor
              case/train/step: refactor
              case/train/stages: vanilla
              case/train/optimizer: lbfgs
              case/train/loss: tik
              case/train: train
              case/plt: plt
              case/data/postprocess: simple
              case/data/preprocess: constrained_vel
              case/data: data
              hydra/env: default
              hydra/callbacks: null
              hydra/job_logging: default
              hydra/hydra_logging: default
              hydra/hydra_help: default
              hydra/help: default
              hydra/sweeper: basic
              hydra/launcher: basic
              hydra/output: default
          verbose: false


    .. admonition:: resolved_config.yaml
      :class: toggle


      .. code-block:: yaml

        case: 
          data: 
            path: conda/data/marmousi/deepwave_example/shots16
            postprocess: 
              __call__: ^^null|misfit_toys.beta.postprocess|vp_compare
              kw: 
                max_iters: 25
                name: tik
                proj_path: marmousi/deepwave_example/shots16
            prefix: conda/data
            preprocess: 
              chunk_keys: 
                params:
                - src_amp_y
                tensors:
                - obs_data
                - src_loc_y
                - rec_loc_y
              dep: ^^null|misfit_toys.fwi.seismic_data|null
              maxv: 2500
              minv: 1000
              path_builder_kw: 
                obs_data: null
                rec_loc_y: null
                remap: 
                  vp_init: vp
                src_amp_y: 
                  kw: 
                    requires_grad: false
                  runtime_func: self.data.preprocess.dep.Param.delay_init
                src_loc_y: null
                vp_init: 
                  kw: 
                    maxv: 2500
                    minv: 1000
                    requires_grad: true
                  runtime_func: self.data.preprocess.dep.ParamConstrained.delay_init
              required_fields:
              - vp_init
              - src_amp_y
              - obs_data
              - src_loc_y
              - rec_loc_y
              - meta
              time_pad_frac: 0.2
            proj_path: marmousi/deepwave_example/shots16
          dupe: true
          editor: code
          name: tik
          np: self.runtime.data.meta.nt
          plt: 
            trace: 
              color_seq:
              - red
              - blue
              iter: 
                none_dims:
                - 0
                - -1
              legend: 
                framealpha: 0.5
                loc: upper right
              linestyles:
              - solid
              - dashed
              save: 
                duration: 250
                path: figs/random_traces.gif
              sub: 
                kw: 
                  figsize:
                  - 10
                  - 10
                shape:
                - 2
                - 2
              suptitle: Observed Data at Random Receiver Locations
              title: Observed Data at Receiver Location
              xlabel: Time (s)
              ylabel: Displacement (m)
            vp: 
              iter: 
                none_dims:
                - -2
                - -1
              order:
              - vp
              - vp_true
              - rel_diff
              plts: 
                rel_diff: 
                  main: 
                    colorbar: true
                    filt: transpose
                    opts: 
                      aspect: auto
                      cmap: seismic
                    title: Relative Difference (%)
                    type: imshow
                    xlabel: Rec Location (m)
                    ylabel: Depth (m)
                vp: 
                  main: 
                    colorbar: true
                    filt: 'eval(lambda x : x.T)'
                    opts: 
                      aspect: auto
                      cmap: seismic
                    title: $v_p$
                    type: imshow
                    xlabel: Rec Location (m)
                    ylabel: Depth (m)
                vp_true: 
                  main: 
                    colorbar: true
                    filt: transpose
                    opts: 
                      aspect: auto
                      cmap: seismic
                    title: $v_{true}$
                    type: imshow
                    xlabel: Rec Location (m)
                    ylabel: Depth (m)
              save: 
                duration: 250
                movie_format: gif
                path: figs/vp.gif
              sub: 
                adjust: 
                  hspace: 0.5
                  wspace: 0.5
                kw: 
                  figsize:
                  - 10
                  - 10
                shape:
                - 2
                - 2
          port: 12576
          train: 
            loss: 
              dep: 
                mod: ^^misfit_toys.fwi.loss.tikhonov
              kw: 
                kw: 
                  _min: 1.0e-07
                  max_iters: 25
                  scale: 1.0e-06
                  weights: self.runtime.prop.module.vp
                runtime_func: self.train.loss.dep.mod.lin_reg_drop
              runtime_func: self.train.loss.dep.mod.TikhonovLoss
            max_iters: 25
            optimizer: 
              args: []
              kw: 
                history_size: 100
                line_search_fn: null
                lr: 1.0
                max_eval: null
                max_iter: 20
                tolerance_change: 1.0e-09
                tolerance_grad: 1.0e-07
              runtime_func: 'eval(lambda *args, **kw: [torch.optim.LBFGS, kw])'
            prop: 
              kw: 
                dt: self.runtime.data.meta.dt
                dx: self.runtime.data.meta.dx
                freq: self.runtime.data.meta.freq
                rec_loc_y: self.runtime.data.rec_loc_y
                src_loc_y: self.runtime.data.src_loc_y
                vp: self.runtime.data.vp
              runtime_func: ^^|misfit_toys.fwi.seismic_data|DebugProp
            retrain: true
            stages: 
              kw: 
                max_iters: 25
              runtime_func: ^^null|misfit_toys.workflows.stages|vanilla_stages
            step: 
              kw: 
                length: 100
                num_batches: null
                scale: 1000000.0
                src_amp_y: self.runtime.data.src_amp_y
              nonexist: checking_the_yaml_format_hook
              runtime_func: ^^null|misfit_toys.workflows.tik.steps|taper_only
        run: test


vp
--

.. image:: figs/vp.gif
   :align: center

random_traces
-------------

.. image:: figs/random_traces.gif
   :align: center
