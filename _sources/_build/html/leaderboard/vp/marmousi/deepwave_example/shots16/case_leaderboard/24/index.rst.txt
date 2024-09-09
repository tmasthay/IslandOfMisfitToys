24
==

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

        human_timestamp: June 10, 2024 10:41:09 PM
        l2_diff: 64.95635986328125
        max_iters: 5
        name: Tikhonov Regularization
        orig_root: /home/tyler/Documents/repos/IslandOfMisfitToys/misfit_toys/hydra/multirun/HYDRA_TIME_2024-06-10/HYDRA_TIME_22-41-09/marmousi/deepwave_example/shots16/5/
        proj_path: marmousi/deepwave_example/shots16
        root: IslandOfMisfitToys/leaderboard/vp/marmousi/deepwave_example/shots16/case_leaderboard/42
        timestamp: 2024-06-10 22-41-09
        train_time: 60.2958083152771


  .. admonition:: hyperparameters
    :class: toggle


    .. admonition:: config.yaml
      :class: toggle


      .. code-block:: yaml

        case:
          port: 12576
          dupe: false
          editor: code
          name: Tikhonov Regularization
          np: self.runtime.prop.module.meta.nt
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
            max_iters: 4
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
              nonexist: checking_the_yaml_format_hook
        run: test


    .. admonition:: overrides.yaml
      :class: toggle


      .. code-block:: yaml

        - +run=test
        - case.train.max_iters=4
        - case.dupe=False


  .. admonition:: version control
    :class: toggle


    .. admonition:: git_info.txt
      :class: toggle


      .. code-block:: text

        HASH: ec935f1d9a5b44b019b8a028874c142b249caf8c
        BRANCH: devel
        
        UNTRACKED FILES: .latest_run
        out/loss_record.pt
        out/loss_record_0.pt
        out/loss_record_1.pt
        out/out_record.pt
        out/out_record_0.pt
        out/out_record_1.pt
        out/vp_record.pt
        out/vp_record_0.pt
        out/vp_record_1.pt
        tmp.txt
        
        ********************************************************************************
        DIFF: diff --git a/docs/meta/leaderboard.py b/docs/meta/leaderboard.py
        index 4485a38..6a9a8cb 100644
        --- a/docs/meta/leaderboard.py
        +++ b/docs/meta/leaderboard.py
        @@ -306,7 +306,7 @@ def make_leaderboard_page(
             data = []
             for i, subdir in enumerate(subdirs):
                 d = [i + 1]
        -        subsubdirs = next(os.walk(pjoin(path, subdir)))[1]
        +        subsubdirs = list(os.walk(pjoin(path, subdir)))[0][1]
                 matches = [e for e in subsubdirs if re.match(r'.*compare.yaml', e)]
                 if len(matches) != 1:
                     raise ValueError(
        @@ -471,6 +471,7 @@ def centralize_info(*, paths, param, score, leaderboard_size, idx_gen):
                 for line in lines:
                     with open(line, 'r') as f:
                         meta = yaml.load(f, Loader=yaml.FullLoader)
        +                meta['root'] = os.path.dirname(line)
                         if meta['proj_path'] in reg_dict:
                             reg_dict[meta['proj_path']].append(meta)
                         else:
        @@ -495,7 +496,7 @@ def centralize_info(*, paths, param, score, leaderboard_size, idx_gen):
                     for rank, e in enumerate(v):
                         # curr_dump_path = pjoin(dump_path, str(rank + 1))
                         curr_dump_path = pjoin(dump_path, str(rank))
        -                input(f'{e["root"]=} {curr_dump_path=}')
        +                # input(f'{e["root"]=} {curr_dump_path=}')
                         os.system(f"cp -r {e['root']} {curr_dump_path}")
                         with open(
                             pjoin(curr_dump_path, f'{param}_compare.yaml'), 'w'
        diff --git a/misfit_toys/beta/postprocess.py b/misfit_toys/beta/postprocess.py
        index 2220da4..0572e82 100644
        --- a/misfit_toys/beta/postprocess.py
        +++ b/misfit_toys/beta/postprocess.py
        @@ -24,7 +24,7 @@ def core_meta(
             *, path: str, proj_path: str, train_time: float, name: str
         ) -> dict:
             meta = get_timestamps(path)
        -    meta['root'] = path
        +    meta['orig_root'] = path
             meta['proj_path'] = proj_path
             meta['train_time'] = train_time
             meta['name'] = name
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
          mode: MULTIRUN
          searchpath: []
          callbacks: {}
          output_subdir: ''
          overrides:
            hydra:
            - hydra.mode=MULTIRUN
            task:
            - +run=test
            - case.train.max_iters=4
            - case.dupe=False
          job:
            name: main
            chdir: null
            override_dirname: +run=test,case.dupe=False,case.train.max_iters=4
            id: '5'
            num: 5
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
            output_dir: /home/tyler/Documents/repos/IslandOfMisfitToys/misfit_toys/hydra/multirun/HYDRA_TIME_2024-06-10/HYDRA_TIME_22-41-09/marmousi/deepwave_example/shots16/5
            choices:
              case: tik
              case/train/step: tik
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


    .. admonition:: index.rst
      :class: toggle


      .. code-block:: text

        42
        ==
        
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
        
                human_timestamp: June 10, 2024 10:41:09 PM
                l2_diff: 64.95635986328125
                max_iters: 5
                name: Tikhonov Regularization
                orig_root: /home/tyler/Documents/repos/IslandOfMisfitToys/misfit_toys/hydra/multirun/HYDRA_TIME_2024-06-10/HYDRA_TIME_22-41-09/marmousi/deepwave_example/shots16/5/
                proj_path: marmousi/deepwave_example/shots16
                root: IslandOfMisfitToys/leaderboard/vp/marmousi/deepwave_example/shots16/case_leaderboard/24
                timestamp: 2024-06-10 22-41-09
                train_time: 60.2958083152771
        
        
          .. admonition:: hyperparameters
            :class: toggle
        
        
            .. admonition:: config.yaml
              :class: toggle
        
        
              .. code-block:: yaml
        
                case:
                  port: 12576
                  dupe: false
                  editor: code
                  name: Tikhonov Regularization
                  np: self.runtime.prop.module.meta.nt
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
                    max_iters: 4
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
                      nonexist: checking_the_yaml_format_hook
                run: test
        
        
            .. admonition:: overrides.yaml
              :class: toggle
        
        
              .. code-block:: yaml
        
                - +run=test
                - case.train.max_iters=4
                - case.dupe=False
        
        
          .. admonition:: version control
            :class: toggle
        
        
            .. admonition:: git_info.txt
              :class: toggle
        
        
              .. code-block:: text
        
                HASH: ec935f1d9a5b44b019b8a028874c142b249caf8c
                BRANCH: devel
                
                UNTRACKED FILES: .latest_run
                out/loss_record.pt
                out/loss_record_0.pt
                out/loss_record_1.pt
                out/out_record.pt
                out/out_record_0.pt
                out/out_record_1.pt
                out/vp_record.pt
                out/vp_record_0.pt
                out/vp_record_1.pt
                tmp.txt
                
                ********************************************************************************
                DIFF: diff --git a/docs/meta/leaderboard.py b/docs/meta/leaderboard.py
                index 4485a38..6a9a8cb 100644
                --- a/docs/meta/leaderboard.py
                +++ b/docs/meta/leaderboard.py
                @@ -306,7 +306,7 @@ def make_leaderboard_page(
                     data = []
                     for i, subdir in enumerate(subdirs):
                         d = [i + 1]
                -        subsubdirs = next(os.walk(pjoin(path, subdir)))[1]
                +        subsubdirs = list(os.walk(pjoin(path, subdir)))[0][1]
                         matches = [e for e in subsubdirs if re.match(r'.*compare.yaml', e)]
                         if len(matches) != 1:
                             raise ValueError(
                @@ -471,6 +471,7 @@ def centralize_info(*, paths, param, score, leaderboard_size, idx_gen):
                         for line in lines:
                             with open(line, 'r') as f:
                                 meta = yaml.load(f, Loader=yaml.FullLoader)
                +                meta['root'] = os.path.dirname(line)
                                 if meta['proj_path'] in reg_dict:
                                     reg_dict[meta['proj_path']].append(meta)
                                 else:
                @@ -495,7 +496,7 @@ def centralize_info(*, paths, param, score, leaderboard_size, idx_gen):
                             for rank, e in enumerate(v):
                                 # curr_dump_path = pjoin(dump_path, str(rank + 1))
                                 curr_dump_path = pjoin(dump_path, str(rank))
                -                input(f'{e["root"]=} {curr_dump_path=}')
                +                # input(f'{e["root"]=} {curr_dump_path=}')
                                 os.system(f"cp -r {e['root']} {curr_dump_path}")
                                 with open(
                                     pjoin(curr_dump_path, f'{param}_compare.yaml'), 'w'
                diff --git a/misfit_toys/beta/postprocess.py b/misfit_toys/beta/postprocess.py
                index 2220da4..0572e82 100644
                --- a/misfit_toys/beta/postprocess.py
                +++ b/misfit_toys/beta/postprocess.py
                @@ -24,7 +24,7 @@ def core_meta(
                     *, path: str, proj_path: str, train_time: float, name: str
                 ) -> dict:
                     meta = get_timestamps(path)
                -    meta['root'] = path
                +    meta['orig_root'] = path
                     meta['proj_path'] = proj_path
                     meta['train_time'] = train_time
                     meta['name'] = name
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
                  mode: MULTIRUN
                  searchpath: []
                  callbacks: {}
                  output_subdir: ''
                  overrides:
                    hydra:
                    - hydra.mode=MULTIRUN
                    task:
                    - +run=test
                    - case.train.max_iters=4
                    - case.dupe=False
                  job:
                    name: main
                    chdir: null
                    override_dirname: +run=test,case.dupe=False,case.train.max_iters=4
                    id: '5'
                    num: 5
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
                    output_dir: /home/tyler/Documents/repos/IslandOfMisfitToys/misfit_toys/hydra/multirun/HYDRA_TIME_2024-06-10/HYDRA_TIME_22-41-09/marmousi/deepwave_example/shots16/5
                    choices:
                      case: tik
                      case/train/step: tik
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
        
        
            .. admonition:: index.rst
              :class: toggle
        
        
              .. code-block:: text
        
                24
                ==
                
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
                
                        human_timestamp: June 10, 2024 10:41:09 PM
                        l2_diff: 64.95635986328125
                        max_iters: 5
                        name: Tikhonov Regularization
                        orig_root: /home/tyler/Documents/repos/IslandOfMisfitToys/misfit_toys/hydra/multirun/HYDRA_TIME_2024-06-10/HYDRA_TIME_22-41-09/marmousi/deepwave_example/shots16/5/
                        proj_path: marmousi/deepwave_example/shots16
                        root: IslandOfMisfitToys/leaderboard/vp/marmousi/deepwave_example/shots16/case_leaderboard/15
                        timestamp: 2024-06-10 22-41-09
                        train_time: 60.2958083152771
                
                
                  .. admonition:: hyperparameters
                    :class: toggle
                
                
                    .. admonition:: config.yaml
                      :class: toggle
                
                
                      .. code-block:: yaml
                
                        case:
                          port: 12576
                          dupe: false
                          editor: code
                          name: Tikhonov Regularization
                          np: self.runtime.prop.module.meta.nt
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
                            max_iters: 4
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
                              nonexist: checking_the_yaml_format_hook
                        run: test
                
                
                    .. admonition:: overrides.yaml
                      :class: toggle
                
                
                      .. code-block:: yaml
                
                        - +run=test
                        - case.train.max_iters=4
                        - case.dupe=False
                
                
                  .. admonition:: version control
                    :class: toggle
                
                
                    .. admonition:: git_info.txt
                      :class: toggle
                
                
                      .. code-block:: text
                
                        HASH: ec935f1d9a5b44b019b8a028874c142b249caf8c
                        BRANCH: devel
                        
                        UNTRACKED FILES: .latest_run
                        out/loss_record.pt
                        out/loss_record_0.pt
                        out/loss_record_1.pt
                        out/out_record.pt
                        out/out_record_0.pt
                        out/out_record_1.pt
                        out/vp_record.pt
                        out/vp_record_0.pt
                        out/vp_record_1.pt
                        tmp.txt
                        
                        ********************************************************************************
                        DIFF: diff --git a/docs/meta/leaderboard.py b/docs/meta/leaderboard.py
                        index 4485a38..6a9a8cb 100644
                        --- a/docs/meta/leaderboard.py
                        +++ b/docs/meta/leaderboard.py
                        @@ -306,7 +306,7 @@ def make_leaderboard_page(
                             data = []
                             for i, subdir in enumerate(subdirs):
                                 d = [i + 1]
                        -        subsubdirs = next(os.walk(pjoin(path, subdir)))[1]
                        +        subsubdirs = list(os.walk(pjoin(path, subdir)))[0][1]
                                 matches = [e for e in subsubdirs if re.match(r'.*compare.yaml', e)]
                                 if len(matches) != 1:
                                     raise ValueError(
                        @@ -471,6 +471,7 @@ def centralize_info(*, paths, param, score, leaderboard_size, idx_gen):
                                 for line in lines:
                                     with open(line, 'r') as f:
                                         meta = yaml.load(f, Loader=yaml.FullLoader)
                        +                meta['root'] = os.path.dirname(line)
                                         if meta['proj_path'] in reg_dict:
                                             reg_dict[meta['proj_path']].append(meta)
                                         else:
                        @@ -495,7 +496,7 @@ def centralize_info(*, paths, param, score, leaderboard_size, idx_gen):
                                     for rank, e in enumerate(v):
                                         # curr_dump_path = pjoin(dump_path, str(rank + 1))
                                         curr_dump_path = pjoin(dump_path, str(rank))
                        -                input(f'{e["root"]=} {curr_dump_path=}')
                        +                # input(f'{e["root"]=} {curr_dump_path=}')
                                         os.system(f"cp -r {e['root']} {curr_dump_path}")
                                         with open(
                                             pjoin(curr_dump_path, f'{param}_compare.yaml'), 'w'
                        diff --git a/misfit_toys/beta/postprocess.py b/misfit_toys/beta/postprocess.py
                        index 2220da4..0572e82 100644
                        --- a/misfit_toys/beta/postprocess.py
                        +++ b/misfit_toys/beta/postprocess.py
                        @@ -24,7 +24,7 @@ def core_meta(
                             *, path: str, proj_path: str, train_time: float, name: str
                         ) -> dict:
                             meta = get_timestamps(path)
                        -    meta['root'] = path
                        +    meta['orig_root'] = path
                             meta['proj_path'] = proj_path
                             meta['train_time'] = train_time
                             meta['name'] = name
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
                          mode: MULTIRUN
                          searchpath: []
                          callbacks: {}
                          output_subdir: ''
                          overrides:
                            hydra:
                            - hydra.mode=MULTIRUN
                            task:
                            - +run=test
                            - case.train.max_iters=4
                            - case.dupe=False
                          job:
                            name: main
                            chdir: null
                            override_dirname: +run=test,case.dupe=False,case.train.max_iters=4
                            id: '5'
                            num: 5
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
                            output_dir: /home/tyler/Documents/repos/IslandOfMisfitToys/misfit_toys/hydra/multirun/HYDRA_TIME_2024-06-10/HYDRA_TIME_22-41-09/marmousi/deepwave_example/shots16/5
                            choices:
                              case: tik
                              case/train/step: tik
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
                
                
                    .. admonition:: index.rst
                      :class: toggle
                
                
                      .. code-block:: text
                
                        15
                        ==
                        
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
                        
                                human_timestamp: June 10, 2024 10:41:09 PM
                                l2_diff: 64.95635986328125
                                max_iters: 5
                                name: Tikhonov Regularization
                                orig_root: /home/tyler/Documents/repos/IslandOfMisfitToys/misfit_toys/hydra/multirun/HYDRA_TIME_2024-06-10/HYDRA_TIME_22-41-09/marmousi/deepwave_example/shots16/5/
                                proj_path: marmousi/deepwave_example/shots16
                                root: IslandOfMisfitToys/leaderboard/vp/marmousi/deepwave_example/shots16/case_leaderboard/6
                                timestamp: 2024-06-10 22-41-09
                                train_time: 60.2958083152771
                        
                        
                          .. admonition:: hyperparameters
                            :class: toggle
                        
                        
                            .. admonition:: config.yaml
                              :class: toggle
                        
                        
                              .. code-block:: yaml
                        
                                case:
                                  port: 12576
                                  dupe: false
                                  editor: code
                                  name: Tikhonov Regularization
                                  np: self.runtime.prop.module.meta.nt
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
                                    max_iters: 4
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
                                      nonexist: checking_the_yaml_format_hook
                                run: test
                        
                        
                            .. admonition:: overrides.yaml
                              :class: toggle
                        
                        
                              .. code-block:: yaml
                        
                                - +run=test
                                - case.train.max_iters=4
                                - case.dupe=False
                        
                        
                          .. admonition:: version control
                            :class: toggle
                        
                        
                            .. admonition:: git_info.txt
                              :class: toggle
                        
                        
                              .. code-block:: text
                        
                                HASH: ec935f1d9a5b44b019b8a028874c142b249caf8c
                                BRANCH: devel
                                
                                UNTRACKED FILES: .latest_run
                                out/loss_record.pt
                                out/loss_record_0.pt
                                out/loss_record_1.pt
                                out/out_record.pt
                                out/out_record_0.pt
                                out/out_record_1.pt
                                out/vp_record.pt
                                out/vp_record_0.pt
                                out/vp_record_1.pt
                                tmp.txt
                                
                                ********************************************************************************
                                DIFF: diff --git a/docs/meta/leaderboard.py b/docs/meta/leaderboard.py
                                index 4485a38..6a9a8cb 100644
                                --- a/docs/meta/leaderboard.py
                                +++ b/docs/meta/leaderboard.py
                                @@ -306,7 +306,7 @@ def make_leaderboard_page(
                                     data = []
                                     for i, subdir in enumerate(subdirs):
                                         d = [i + 1]
                                -        subsubdirs = next(os.walk(pjoin(path, subdir)))[1]
                                +        subsubdirs = list(os.walk(pjoin(path, subdir)))[0][1]
                                         matches = [e for e in subsubdirs if re.match(r'.*compare.yaml', e)]
                                         if len(matches) != 1:
                                             raise ValueError(
                                @@ -471,6 +471,7 @@ def centralize_info(*, paths, param, score, leaderboard_size, idx_gen):
                                         for line in lines:
                                             with open(line, 'r') as f:
                                                 meta = yaml.load(f, Loader=yaml.FullLoader)
                                +                meta['root'] = os.path.dirname(line)
                                                 if meta['proj_path'] in reg_dict:
                                                     reg_dict[meta['proj_path']].append(meta)
                                                 else:
                                @@ -495,7 +496,7 @@ def centralize_info(*, paths, param, score, leaderboard_size, idx_gen):
                                             for rank, e in enumerate(v):
                                                 # curr_dump_path = pjoin(dump_path, str(rank + 1))
                                                 curr_dump_path = pjoin(dump_path, str(rank))
                                -                input(f'{e["root"]=} {curr_dump_path=}')
                                +                # input(f'{e["root"]=} {curr_dump_path=}')
                                                 os.system(f"cp -r {e['root']} {curr_dump_path}")
                                                 with open(
                                                     pjoin(curr_dump_path, f'{param}_compare.yaml'), 'w'
                                diff --git a/misfit_toys/beta/postprocess.py b/misfit_toys/beta/postprocess.py
                                index 2220da4..0572e82 100644
                                --- a/misfit_toys/beta/postprocess.py
                                +++ b/misfit_toys/beta/postprocess.py
                                @@ -24,7 +24,7 @@ def core_meta(
                                     *, path: str, proj_path: str, train_time: float, name: str
                                 ) -> dict:
                                     meta = get_timestamps(path)
                                -    meta['root'] = path
                                +    meta['orig_root'] = path
                                     meta['proj_path'] = proj_path
                                     meta['train_time'] = train_time
                                     meta['name'] = name
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
                                  mode: MULTIRUN
                                  searchpath: []
                                  callbacks: {}
                                  output_subdir: ''
                                  overrides:
                                    hydra:
                                    - hydra.mode=MULTIRUN
                                    task:
                                    - +run=test
                                    - case.train.max_iters=4
                                    - case.dupe=False
                                  job:
                                    name: main
                                    chdir: null
                                    override_dirname: +run=test,case.dupe=False,case.train.max_iters=4
                                    id: '5'
                                    num: 5
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
                                    output_dir: /home/tyler/Documents/repos/IslandOfMisfitToys/misfit_toys/hydra/multirun/HYDRA_TIME_2024-06-10/HYDRA_TIME_22-41-09/marmousi/deepwave_example/shots16/5
                                    choices:
                                      case: tik
                                      case/train/step: tik
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
                        
                        
                            .. admonition:: index.rst
                              :class: toggle
                        
                        
                              .. code-block:: text
                        
                                6
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
                                
                                        human_timestamp: June 10, 2024 10:41:09 PM
                                        l2_diff: 64.95635986328125
                                        max_iters: 5
                                        name: Tikhonov Regularization
                                        orig_root: /home/tyler/Documents/repos/IslandOfMisfitToys/misfit_toys/hydra/multirun/HYDRA_TIME_2024-06-10/HYDRA_TIME_22-41-09/marmousi/deepwave_example/shots16/5/
                                        proj_path: marmousi/deepwave_example/shots16
                                        root: IslandOfMisfitToys/leaderboard/vp/marmousi/deepwave_example/shots16/case_leaderboard/5
                                        timestamp: 2024-06-10 22-41-09
                                        train_time: 60.2958083152771
                                
                                
                                  .. admonition:: hyperparameters
                                    :class: toggle
                                
                                
                                    .. admonition:: config.yaml
                                      :class: toggle
                                
                                
                                      .. code-block:: yaml
                                
                                        case:
                                          port: 12576
                                          dupe: false
                                          editor: code
                                          name: Tikhonov Regularization
                                          np: self.runtime.prop.module.meta.nt
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
                                            max_iters: 4
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
                                              nonexist: checking_the_yaml_format_hook
                                        run: test
                                
                                
                                    .. admonition:: overrides.yaml
                                      :class: toggle
                                
                                
                                      .. code-block:: yaml
                                
                                        - +run=test
                                        - case.train.max_iters=4
                                        - case.dupe=False
                                
                                
                                  .. admonition:: version control
                                    :class: toggle
                                
                                
                                    .. admonition:: git_info.txt
                                      :class: toggle
                                
                                
                                      .. code-block:: text
                                
                                        HASH: ec935f1d9a5b44b019b8a028874c142b249caf8c
                                        BRANCH: devel
                                        
                                        UNTRACKED FILES: .latest_run
                                        out/loss_record.pt
                                        out/loss_record_0.pt
                                        out/loss_record_1.pt
                                        out/out_record.pt
                                        out/out_record_0.pt
                                        out/out_record_1.pt
                                        out/vp_record.pt
                                        out/vp_record_0.pt
                                        out/vp_record_1.pt
                                        tmp.txt
                                        
                                        ********************************************************************************
                                        DIFF: diff --git a/docs/meta/leaderboard.py b/docs/meta/leaderboard.py
                                        index 4485a38..6a9a8cb 100644
                                        --- a/docs/meta/leaderboard.py
                                        +++ b/docs/meta/leaderboard.py
                                        @@ -306,7 +306,7 @@ def make_leaderboard_page(
                                             data = []
                                             for i, subdir in enumerate(subdirs):
                                                 d = [i + 1]
                                        -        subsubdirs = next(os.walk(pjoin(path, subdir)))[1]
                                        +        subsubdirs = list(os.walk(pjoin(path, subdir)))[0][1]
                                                 matches = [e for e in subsubdirs if re.match(r'.*compare.yaml', e)]
                                                 if len(matches) != 1:
                                                     raise ValueError(
                                        @@ -471,6 +471,7 @@ def centralize_info(*, paths, param, score, leaderboard_size, idx_gen):
                                                 for line in lines:
                                                     with open(line, 'r') as f:
                                                         meta = yaml.load(f, Loader=yaml.FullLoader)
                                        +                meta['root'] = os.path.dirname(line)
                                                         if meta['proj_path'] in reg_dict:
                                                             reg_dict[meta['proj_path']].append(meta)
                                                         else:
                                        @@ -495,7 +496,7 @@ def centralize_info(*, paths, param, score, leaderboard_size, idx_gen):
                                                     for rank, e in enumerate(v):
                                                         # curr_dump_path = pjoin(dump_path, str(rank + 1))
                                                         curr_dump_path = pjoin(dump_path, str(rank))
                                        -                input(f'{e["root"]=} {curr_dump_path=}')
                                        +                # input(f'{e["root"]=} {curr_dump_path=}')
                                                         os.system(f"cp -r {e['root']} {curr_dump_path}")
                                                         with open(
                                                             pjoin(curr_dump_path, f'{param}_compare.yaml'), 'w'
                                        diff --git a/misfit_toys/beta/postprocess.py b/misfit_toys/beta/postprocess.py
                                        index 2220da4..0572e82 100644
                                        --- a/misfit_toys/beta/postprocess.py
                                        +++ b/misfit_toys/beta/postprocess.py
                                        @@ -24,7 +24,7 @@ def core_meta(
                                             *, path: str, proj_path: str, train_time: float, name: str
                                         ) -> dict:
                                             meta = get_timestamps(path)
                                        -    meta['root'] = path
                                        +    meta['orig_root'] = path
                                             meta['proj_path'] = proj_path
                                             meta['train_time'] = train_time
                                             meta['name'] = name
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
                                          mode: MULTIRUN
                                          searchpath: []
                                          callbacks: {}
                                          output_subdir: ''
                                          overrides:
                                            hydra:
                                            - hydra.mode=MULTIRUN
                                            task:
                                            - +run=test
                                            - case.train.max_iters=4
                                            - case.dupe=False
                                          job:
                                            name: main
                                            chdir: null
                                            override_dirname: +run=test,case.dupe=False,case.train.max_iters=4
                                            id: '5'
                                            num: 5
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
                                            output_dir: /home/tyler/Documents/repos/IslandOfMisfitToys/misfit_toys/hydra/multirun/HYDRA_TIME_2024-06-10/HYDRA_TIME_22-41-09/marmousi/deepwave_example/shots16/5
                                            choices:
                                              case: tik
                                              case/train/step: tik
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
                                
                                
                                    .. admonition:: index.rst
                                      :class: toggle
                                
                                
                                      .. code-block:: text
                                
                                        5
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
                                        
                                                human_timestamp: June 10, 2024 10:41:09 PM
                                                l2_diff: 64.95635986328125
                                                max_iters: 5
                                                name: Tikhonov Regularization
                                                orig_root: /home/tyler/Documents/repos/IslandOfMisfitToys/misfit_toys/hydra/multirun/HYDRA_TIME_2024-06-10/HYDRA_TIME_22-41-09/marmousi/deepwave_example/shots16/5/
                                                proj_path: marmousi/deepwave_example/shots16
                                                root: ../../misfit_toys/hydra/multirun/HYDRA_TIME_2024-06-10/HYDRA_TIME_22-41-09/marmousi/deepwave_example/shots16/5
                                                timestamp: 2024-06-10 22-41-09
                                                train_time: 60.2958083152771
                                        
                                        
                                          .. admonition:: hyperparameters
                                            :class: toggle
                                        
                                        
                                            .. admonition:: config.yaml
                                              :class: toggle
                                        
                                        
                                              .. code-block:: yaml
                                        
                                                case:
                                                  port: 12576
                                                  dupe: false
                                                  editor: code
                                                  name: Tikhonov Regularization
                                                  np: self.runtime.prop.module.meta.nt
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
                                                    max_iters: 4
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
                                                      nonexist: checking_the_yaml_format_hook
                                                run: test
                                        
                                        
                                            .. admonition:: overrides.yaml
                                              :class: toggle
                                        
                                        
                                              .. code-block:: yaml
                                        
                                                - +run=test
                                                - case.train.max_iters=4
                                                - case.dupe=False
                                        
                                        
                                          .. admonition:: version control
                                            :class: toggle
                                        
                                        
                                            .. admonition:: git_info.txt
                                              :class: toggle
                                        
                                        
                                              .. code-block:: text
                                        
                                                HASH: ec935f1d9a5b44b019b8a028874c142b249caf8c
                                                BRANCH: devel
                                                
                                                UNTRACKED FILES: .latest_run
                                                out/loss_record.pt
                                                out/loss_record_0.pt
                                                out/loss_record_1.pt
                                                out/out_record.pt
                                                out/out_record_0.pt
                                                out/out_record_1.pt
                                                out/vp_record.pt
                                                out/vp_record_0.pt
                                                out/vp_record_1.pt
                                                tmp.txt
                                                
                                                ********************************************************************************
                                                DIFF: diff --git a/docs/meta/leaderboard.py b/docs/meta/leaderboard.py
                                                index 4485a38..6a9a8cb 100644
                                                --- a/docs/meta/leaderboard.py
                                                +++ b/docs/meta/leaderboard.py
                                                @@ -306,7 +306,7 @@ def make_leaderboard_page(
                                                     data = []
                                                     for i, subdir in enumerate(subdirs):
                                                         d = [i + 1]
                                                -        subsubdirs = next(os.walk(pjoin(path, subdir)))[1]
                                                +        subsubdirs = list(os.walk(pjoin(path, subdir)))[0][1]
                                                         matches = [e for e in subsubdirs if re.match(r'.*compare.yaml', e)]
                                                         if len(matches) != 1:
                                                             raise ValueError(
                                                @@ -471,6 +471,7 @@ def centralize_info(*, paths, param, score, leaderboard_size, idx_gen):
                                                         for line in lines:
                                                             with open(line, 'r') as f:
                                                                 meta = yaml.load(f, Loader=yaml.FullLoader)
                                                +                meta['root'] = os.path.dirname(line)
                                                                 if meta['proj_path'] in reg_dict:
                                                                     reg_dict[meta['proj_path']].append(meta)
                                                                 else:
                                                @@ -495,7 +496,7 @@ def centralize_info(*, paths, param, score, leaderboard_size, idx_gen):
                                                             for rank, e in enumerate(v):
                                                                 # curr_dump_path = pjoin(dump_path, str(rank + 1))
                                                                 curr_dump_path = pjoin(dump_path, str(rank))
                                                -                input(f'{e["root"]=} {curr_dump_path=}')
                                                +                # input(f'{e["root"]=} {curr_dump_path=}')
                                                                 os.system(f"cp -r {e['root']} {curr_dump_path}")
                                                                 with open(
                                                                     pjoin(curr_dump_path, f'{param}_compare.yaml'), 'w'
                                                diff --git a/misfit_toys/beta/postprocess.py b/misfit_toys/beta/postprocess.py
                                                index 2220da4..0572e82 100644
                                                --- a/misfit_toys/beta/postprocess.py
                                                +++ b/misfit_toys/beta/postprocess.py
                                                @@ -24,7 +24,7 @@ def core_meta(
                                                     *, path: str, proj_path: str, train_time: float, name: str
                                                 ) -> dict:
                                                     meta = get_timestamps(path)
                                                -    meta['root'] = path
                                                +    meta['orig_root'] = path
                                                     meta['proj_path'] = proj_path
                                                     meta['train_time'] = train_time
                                                     meta['name'] = name
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
                                                  mode: MULTIRUN
                                                  searchpath: []
                                                  callbacks: {}
                                                  output_subdir: ''
                                                  overrides:
                                                    hydra:
                                                    - hydra.mode=MULTIRUN
                                                    task:
                                                    - +run=test
                                                    - case.train.max_iters=4
                                                    - case.dupe=False
                                                  job:
                                                    name: main
                                                    chdir: null
                                                    override_dirname: +run=test,case.dupe=False,case.train.max_iters=4
                                                    id: '5'
                                                    num: 5
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
                                                    output_dir: /home/tyler/Documents/repos/IslandOfMisfitToys/misfit_toys/hydra/multirun/HYDRA_TIME_2024-06-10/HYDRA_TIME_22-41-09/marmousi/deepwave_example/shots16/5
                                                    choices:
                                                      case: tik
                                                      case/train/step: tik
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
                                                        name: Tikhonov Regularization
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
                                                  dupe: false
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
                                                          max_iters: 4
                                                          scale: 1.0e-06
                                                          weights: self.runtime.prop.module.vp
                                                        runtime_func: self.train.loss.dep.mod.lin_reg_drop
                                                      runtime_func: self.train.loss.dep.mod.TikhonovLoss
                                                    max_iters: 4
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
                                                    retrain: true
                                                    stages: 
                                                      kw: 
                                                        max_iters: 4
                                                      runtime_func: ^^null|misfit_toys.workflows.stages|vanilla_stages
                                                    step: 
                                                      kw: 
                                                        length: 100
                                                        num_batches: null
                                                        scale: 1000000.0
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
                                
                                
                                    .. admonition:: resolved_config.yaml
                                      :class: toggle
                                
                                
                                      .. code-block:: yaml
                                
                                        case: 
                                          data: 
                                            path: conda/data/marmousi/deepwave_example/shots16
                                            postprocess: 
                                              __call__: ^^null|misfit_toys.beta.postprocess|vp_compare
                                              kw: 
                                                name: Tikhonov Regularization
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
                                          dupe: false
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
                                                  max_iters: 4
                                                  scale: 1.0e-06
                                                  weights: self.runtime.prop.module.vp
                                                runtime_func: self.train.loss.dep.mod.lin_reg_drop
                                              runtime_func: self.train.loss.dep.mod.TikhonovLoss
                                            max_iters: 4
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
                                            retrain: true
                                            stages: 
                                              kw: 
                                                max_iters: 4
                                              runtime_func: ^^null|misfit_toys.workflows.stages|vanilla_stages
                                            step: 
                                              kw: 
                                                length: 100
                                                num_batches: null
                                                scale: 1000000.0
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
                        
                        
                            .. admonition:: resolved_config.yaml
                              :class: toggle
                        
                        
                              .. code-block:: yaml
                        
                                case: 
                                  data: 
                                    path: conda/data/marmousi/deepwave_example/shots16
                                    postprocess: 
                                      __call__: ^^null|misfit_toys.beta.postprocess|vp_compare
                                      kw: 
                                        name: Tikhonov Regularization
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
                                  dupe: false
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
                                          max_iters: 4
                                          scale: 1.0e-06
                                          weights: self.runtime.prop.module.vp
                                        runtime_func: self.train.loss.dep.mod.lin_reg_drop
                                      runtime_func: self.train.loss.dep.mod.TikhonovLoss
                                    max_iters: 4
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
                                    retrain: true
                                    stages: 
                                      kw: 
                                        max_iters: 4
                                      runtime_func: ^^null|misfit_toys.workflows.stages|vanilla_stages
                                    step: 
                                      kw: 
                                        length: 100
                                        num_batches: null
                                        scale: 1000000.0
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
                
                
                    .. admonition:: resolved_config.yaml
                      :class: toggle
                
                
                      .. code-block:: yaml
                
                        case: 
                          data: 
                            path: conda/data/marmousi/deepwave_example/shots16
                            postprocess: 
                              __call__: ^^null|misfit_toys.beta.postprocess|vp_compare
                              kw: 
                                name: Tikhonov Regularization
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
                          dupe: false
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
                                  max_iters: 4
                                  scale: 1.0e-06
                                  weights: self.runtime.prop.module.vp
                                runtime_func: self.train.loss.dep.mod.lin_reg_drop
                              runtime_func: self.train.loss.dep.mod.TikhonovLoss
                            max_iters: 4
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
                            retrain: true
                            stages: 
                              kw: 
                                max_iters: 4
                              runtime_func: ^^null|misfit_toys.workflows.stages|vanilla_stages
                            step: 
                              kw: 
                                length: 100
                                num_batches: null
                                scale: 1000000.0
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
        
        
            .. admonition:: resolved_config.yaml
              :class: toggle
        
        
              .. code-block:: yaml
        
                case: 
                  data: 
                    path: conda/data/marmousi/deepwave_example/shots16
                    postprocess: 
                      __call__: ^^null|misfit_toys.beta.postprocess|vp_compare
                      kw: 
                        name: Tikhonov Regularization
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
                  dupe: false
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
                          max_iters: 4
                          scale: 1.0e-06
                          weights: self.runtime.prop.module.vp
                        runtime_func: self.train.loss.dep.mod.lin_reg_drop
                      runtime_func: self.train.loss.dep.mod.TikhonovLoss
                    max_iters: 4
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
                    retrain: true
                    stages: 
                      kw: 
                        max_iters: 4
                      runtime_func: ^^null|misfit_toys.workflows.stages|vanilla_stages
                    step: 
                      kw: 
                        length: 100
                        num_batches: null
                        scale: 1000000.0
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


    .. admonition:: resolved_config.yaml
      :class: toggle


      .. code-block:: yaml

        case: 
          data: 
            path: conda/data/marmousi/deepwave_example/shots16
            postprocess: 
              __call__: ^^null|misfit_toys.beta.postprocess|vp_compare
              kw: 
                name: Tikhonov Regularization
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
          dupe: false
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
                  max_iters: 4
                  scale: 1.0e-06
                  weights: self.runtime.prop.module.vp
                runtime_func: self.train.loss.dep.mod.lin_reg_drop
              runtime_func: self.train.loss.dep.mod.TikhonovLoss
            max_iters: 4
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
            retrain: true
            stages: 
              kw: 
                max_iters: 4
              runtime_func: ^^null|misfit_toys.workflows.stages|vanilla_stages
            step: 
              kw: 
                length: 100
                num_batches: null
                scale: 1000000.0
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
