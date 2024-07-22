import hydra
from mh.core import DotDict, set_print_options, torch_stats
from omegaconf import OmegaConf

from misfit_toys.utils import exec_imports, runtime_reduce

set_print_options(callback=torch_stats('all'))


@hydra.main(config_path="cfg", config_name="cfg", version_base=None)
def main(cfg):
    c = DotDict(OmegaConf.to_container(cfg, resolve=True))
    c = exec_imports(c)

    def runtime_reduce_simple(key):
        resolve_rules = c[key].get('resolve', c.resolve)
        c[key] = runtime_reduce(c[key], **resolve_rules, self_key=f'slf_{key}')

    runtime_reduce_simple('vp')
    runtime_reduce_simple('rec_loc_y')
    runtime_reduce_simple('src_loc_y')
    runtime_reduce_simple('src_amp_y')
    runtime_reduce_simple('gbl_rec_loc')
    # runtime_reduce_simple('gbl_obs_data')
    c = runtime_reduce(c, **c.resolve, self_key='slf_gbl_obs_data')

    print(c)


if __name__ == "__main__":
    main()
