8
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

        human_timestamp: June 17, 2024 01:49:15 AM
        l2_diff: 451.92156982421875
        max_iters: 25
        name: Tikhonov Regularization
        orig_root: /home/tyler/Documents/repos/IslandOfMisfitToys/misfit_toys/hydra/outputs/HYDRA_TIME_2024-06-17/HYDRA_TIME_01-49-15/marmousi/
        proj_path: marmousi
        root: ../../misfit_toys/hydra/outputs/HYDRA_TIME_2024-06-17/HYDRA_TIME_01-49-15/marmousi
        timestamp: 2024-06-17 01-49-15
        train_time: 0.0


  .. admonition:: hyperparameters
    :class: toggle


    .. admonition:: config.yaml
      :class: toggle


      .. code-block:: yaml

        case:
          port: 12576
          dupe: true
          editor: code
          name: Tikhonov Regularization
          np: self.runtime.prop.module.meta.nt
          data:
            prefix: conda/data
            proj_path: marmousi
            path: ${.prefix}/${.proj_path}
            preprocess:
              dep: ^^null|misfit_toys.fwi.seismic_data|null
              minv: 1000
              maxv: 5000
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
            retrain: false
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
              runtime_func: ^^null|misfit_toys.workflows.tik.steps|taper_batch
              kw:
                length: 0.3
                batch_size: 4
                scale: 1000000.0
                verbose: true
              nonexist: checking_the_yaml_format_hook
        run: full_test


    .. admonition:: overrides.yaml
      :class: toggle


      .. code-block:: yaml

        - case=tik
        - case/data=full_marmousi
        - case/data/preprocess=full_marmousi
        - case/train/step=tik_full_marmousi
        - +run=full_test
        - case.train.retrain=false


  .. admonition:: version control
    :class: toggle


    .. admonition:: git_info.txt
      :class: toggle


      .. code-block:: text

        HASH: b133f576364bd26a9f933242f93468f076127e7c
        BRANCH: devel
        
        UNTRACKED FILES: .latest_run
        .vscode/settings.json
        out/loss_record.pt
        out/loss_record_0.pt
        out/loss_record_1.pt
        out/out_record.pt
        out/out_record_0.pt
        out/out_record_1.pt
        out/vp_record.pt
        out/vp_record_0.pt
        out/vp_record_1.pt
        
        ********************************************************************************
        DIFF: diff --git a/misfit_toys/examples/hydra/cfg/case/train/step/tik_full_marmousi.yaml b/misfit_toys/examples/hydra/cfg/case/train/step/tik_full_marmousi.yaml
        index 232dbca..2da21d0 100644
        --- a/misfit_toys/examples/hydra/cfg/case/train/step/tik_full_marmousi.yaml
        +++ b/misfit_toys/examples/hydra/cfg/case/train/step/tik_full_marmousi.yaml
        @@ -3,5 +3,6 @@ kw:
           length: 0.3
           batch_size: 4
           scale: 1.0e+06
        +  verbose: true
         
         nonexist: checking_the_yaml_format_hook
        diff --git a/misfit_toys/hydra/main_worker.py b/misfit_toys/hydra/main_worker.py
        index 0566d52..5ee2ba4 100644
        --- a/misfit_toys/hydra/main_worker.py
        +++ b/misfit_toys/hydra/main_worker.py
        @@ -1,5 +1,6 @@
         import os
         from subprocess import check_output as co
        +import sys
         from time import time
         
         import hydra
        @@ -147,6 +148,8 @@ def run_rank(rank: int, world_size: int, c: DotDict) -> None:
         
             optimizer = apply(c.train.optimizer)
             step = apply(c.train.step)
        +    print(str(step))
        +    # sys.exit(1)
             training_stages = apply(c.train.stages)
         
             pre_time = time() - start_pre
        @@ -426,12 +429,17 @@ def main(cfg: DictConfig) -> None:
                 dim=-1,
             )
             rand_indices = [[slice(ee, ee + 1) for ee in e] for e in rand_indices]
        +    input(rand_indices)
        +    input(data.obs_data.shape)
             traces = torch.stack([data.obs_data[s].squeeze() for s in rand_indices])
             out_traces = torch.stack(
                 [data.out[[slice(None), *s]].squeeze() for s in rand_indices]
             )
             d = DotDict({'obs_data': traces, 'out': out_traces})
             trace_iter = bool_slice(*d.out.shape, **c.plt.trace.iter)
        +    input(d.out.shape)
        +    input(c.plt.trace.iter)
        +    input(list(trace_iter))
             fig, axes = plt.subplots(*c.plt.trace.sub.shape, **c.plt.trace.sub.kw)
             trace_frames = get_frames_bool(
                 data=d, iter=trace_iter, fig=fig, axes=axes, plotter=trace_plotter, c=c
        diff --git a/misfit_toys/workflows/tik/steps.py b/misfit_toys/workflows/tik/steps.py
        index 8c00408..7a53d54 100644
        --- a/misfit_toys/workflows/tik/steps.py
        +++ b/misfit_toys/workflows/tik/steps.py
        @@ -1,5 +1,5 @@
         from misfit_toys.utils import taper
        -
        +from time import time
         
         def taper_only(*, length=None, num_batches=None, scale=1.0):
             """
        @@ -40,7 +40,7 @@ def taper_only(*, length=None, num_batches=None, scale=1.0):
             return helper
         
         
        -def taper_batch(*, length=None, batch_size=1, scale=1.0):
        +def taper_batch(*, length=None, batch_size=1, scale=1.0, verbose=True):
             """
             Applies tapering to the output of a neural network model.
         
        @@ -56,9 +56,13 @@ def taper_batch(*, length=None, batch_size=1, scale=1.0):
             Returns:
                 helper (function): A helper function that applies tapering to the output of a neural network model.
             """
        -
        +    num_calls = 0
             def helper(self):
        -        nonlocal length, scale, batch_size
        +        nonlocal length, scale, batch_size, num_calls
        +        start_time = time()
        +        if verbose:
        +            num_calls += 1
        +            print(f"Call {num_calls}...", flush=True, end='')
                 num_shots = self.obs_data.shape[0]
                 num_batches = -(-num_shots // batch_size)
                 slices = [
        @@ -68,6 +72,8 @@ def taper_batch(*, length=None, batch_size=1, scale=1.0):
         
                 self.loss = 0.0
                 for _, s in enumerate(slices):
        +            # if verbose:
        +            #     print(f"Batch {s.start // batch_size + 1}/{num_batches}", flush=True, end='\r')
                     self.out = self.prop(s)[-1]
         
                     if length is not None:
        @@ -81,6 +87,13 @@ def taper_batch(*, length=None, batch_size=1, scale=1.0):
                         obs_data_filt = self.obs_data
         
                     self.loss = scale * self.loss_fn(self.out, obs_data_filt)
        +            # if verbose:
        +            #     print(f"Batch {s.start // batch_size + 1}/{num_batches}: Loss={self.loss}", flush=True)
                     self.loss.backward()
         
        +        if verbose:
        +            total_time = time() - start_time
        +            print(f"took {total_time:.2f} seconds", flush=True)
        +        
        +
             return helper
        ********************************************************************************


  .. admonition:: stdout
    :class: toggle


    .. admonition:: main.log
      :class: toggle


      .. code-block:: text

        Empty file


  .. admonition:: stderr
    :class: toggle


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
            - case=tik
            - case/data=full_marmousi
            - case/data/preprocess=full_marmousi
            - case/train/step=tik_full_marmousi
            - +run=full_test
            - case.train.retrain=false
          job:
            name: main
            chdir: null
            override_dirname: +run=full_test,case.train.retrain=false,case/data/preprocess=full_marmousi,case/data=full_marmousi,case/train/step=tik_full_marmousi,case=tik
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
            output_dir: /home/tyler/Documents/repos/IslandOfMisfitToys/misfit_toys/hydra/outputs/HYDRA_TIME_2024-06-17/HYDRA_TIME_01-49-15/marmousi
            choices:
              case: tik
              case/train/step: tik_full_marmousi
              case/train/stages: vanilla
              case/train/optimizer: lbfgs
              case/train/loss: tik
              case/train: train
              case/plt: plt
              case/data/postprocess: simple
              case/data/preprocess: full_marmousi
              case/data: full_marmousi
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
            path: conda/data/marmousi
            postprocess: 
              __call__: ^^null|misfit_toys.beta.postprocess|vp_compare
              kw: 
                max_iters: 25
                name: Tikhonov Regularization
                proj_path: marmousi
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
              maxv: 5000
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
                    maxv: 5000
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
            proj_path: marmousi
          dupe: true
          editor: code
          name: Tikhonov Regularization
          np: self.runtime.prop.module.meta.nt
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
            retrain: false
            stages: 
              kw: 
                max_iters: 25
              runtime_func: ^^null|misfit_toys.workflows.stages|vanilla_stages
            step: 
              kw: 
                batch_size: 4
                length: 0.3
                scale: 1000000.0
                verbose: true
              nonexist: checking_the_yaml_format_hook
              runtime_func: ^^null|misfit_toys.workflows.tik.steps|taper_batch
        run: full_test

