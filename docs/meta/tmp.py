import os
import re
from os.path import join as pjoin
from subprocess import check_output as co

import hydra
import yaml
from mh.core import convert_dictconfig
from omegaconf import DictConfig


def prune_empty_dirs(*, root, ignore):
    dirs = bottom_up_dirs(root)
    for dir in dirs:
        all_files = sco(f'find {dir} -type f').split('\n')
        all_files = [e for e in all_files if e]
        all_files = [e for e in all_files if not any([i in e for i in ignore])]
        if len(all_files) == 0:
            os.system(f'rm -rf {dir}')
            print(f"prune_empty_dirs: {dir}")


def sco(cmd, verbose=False):
    cmd = ' '.join(cmd.split())
    if verbose:
        print(cmd, flush=True)
    return co(cmd, shell=True).decode().strip()


def bottom_up_dirs(root):
    return sco(f"""
        find {root} -type d |
        awk -F'/' '{{print $0 ": " NF-1}}' |
        sort -t':' -k2,2nr |
        awk -F':' '{{print $1}}'
        """).split('\n')


def get_callback(*, path, idx_gen):
    # note that this works *only* if
    #     default key is *last* in idx_gen
    for k, v in idx_gen.items():
        if k == 'default' or re.search(v['regex'], path):
            f = globals()[v['callback']]
            if 'kw' in v:

                def helper(x):
                    return f(x, **v['kw'])

                return helper
            else:
                return f
    raise ValueError(f"No callback found for path {path}")


def centralize_info(*, paths, param, score, leaderboard_size, idx_gen):
    registered_tests = sco(f"""
        find {paths.src}/data -mindepth 1 -type d |
        grep -v "__pycache__" |
        sed -E 's|{paths.src}/data/||'
        """).split('\n')
    # sort tests by depth of directory
    registered_tests.sort(key=lambda x: x.count('/'), reverse=True)
    reg_dict = {e: [] for e in registered_tests}

    # registered_tests = [e for e in registered_tests if e]
    def get_paths(root, *, peelback, cfg_path):
        nonlocal reg_dict
        lines = sco(f"""
            find {root} -name "{param}_compare.yaml" |
            rev |
            cut -d'/' -f{peelback}- |
            rev |
            awk '{{print $0 "{cfg_path}"}}'
            """).split('\n')
        lines = [e for e in lines if e]
        # input('\n'.join(lines))
        for line in lines:
            og_path = line.replace(cfg_path, '')
            timestamp = ' '.join(
                [e for e in og_path.split('/')[-peelback:] if '-' in e]
            )

            # load the yaml
            cfg = yaml.load(open(line, 'r'), Loader=yaml.FullLoader)
            score_yaml = yaml.load(
                open(
                    pjoin(
                        og_path,
                        (peelback == 3) * 'meta',
                        f"{param}_compare.yaml",
                    ),
                    'r',
                ),
                Loader=yaml.FullLoader,
            )
            score_val = score_yaml[score]
            test_case = cfg['case']['data']['path']
            del cfg, score_yaml
            found_registration = ''
            for reg_test in registered_tests:
                if reg_test in test_case:
                    found_registration = reg_test
                    break
            if not found_registration:
                raise ValueError(
                    f"Test case {test_case} not found in registered tests"
                )
            reg_dict[found_registration].append(
                {'og_path': og_path, 'timestamp': timestamp, 'score': score_val}
            )

    def deploy():
        nonlocal reg_dict
        for k, v in reg_dict.items():
            reg_dict[k] = sorted(v, key=lambda x: float(x['score']))

        # remove duplicates and select out the top leaderboard_size
        for k, v in reg_dict.items():
            unique_items = {tuple(sorted(e.items())): e for e in v}
            reg_dict[k] = list(unique_items.values())
            reg_dict[k] = reg_dict[k][:leaderboard_size]

        for k, v in reg_dict.items():
            root_dump_path = pjoin(paths.final, param, k)
            dump_path = pjoin(root_dump_path, paths.data_dump)
            os.makedirs(root_dump_path, exist_ok=True)
            os.makedirs(dump_path, exist_ok=False)
            for rank, e in enumerate(v):
                curr_dump_path = pjoin(dump_path, str(rank + 1))
                os.system(f"cp -r {e['og_path']} {curr_dump_path}")
                with open(
                    pjoin(curr_dump_path, f'{paths.meta}.yaml'), 'w'
                ) as f:
                    yaml.dump(e, f)
                cmd = (f'''
                    find {curr_dump_path} -type f !
                    -wholename "*figs*" -exec bash -c 'eval "mv $0 {curr_dump_path}"' {{}} \;
                    ''').lstrip()
                cmd = ' '.join(cmd.split())
                os.system(cmd)
                os.system(
                    f'find {curr_dump_path} -mindepth 1 -type d !'
                    ' -wholename "*figs*" -exec rm'
                    ' -rf {} \; 2> /dev/null'
                )

    get_paths(paths.src, peelback=3, cfg_path='/.hydra/config.yaml')
    get_paths(paths.prev_leaders, peelback=2, cfg_path='/config.yaml')
    deploy()

    # all_dirs = bottom_up_dirs(paths.final)
    for dir in bottom_up_dirs(pjoin(paths.final, param)):
        callback = get_callback(path=dir, idx_gen=idx_gen)
        callback(dir)


@hydra.main(config_path="cfg", config_name="cfg", version_base=None)
def main(cfg: DictConfig):
    # TODO: this would be better to refactor with a _templates directory
    c = convert_dictconfig(cfg, self_ref_resolve=False, mutable=False)
    os.system(f'rm -rf {c.git.repo_name}')
    os.system(f'git clone --branch {c.git.branch} --single-branch {c.git.url}')
    os.makedirs(c.rst.dest, exist_ok=True)
    os.system(f'rm -rf {c.folder_name}')
    os.system(f'rm -rf {c.paths.final}')

    # write_folder_structure(
    #     search_root=pjoin(c.paths.src, 'data'),
    #     out_root=c.folder_name,
    #     params=c.params,
    # )
    for param in c.params:
        centralize_info(
            paths=c.paths,
            param=param,
            score=c.score,
            leaderboard_size=c.leaderboard_size,
            idx_gen=c.rst.idx_gen,
        )
    callback = get_callback(path=c.paths.final, idx_gen=c.rst.idx_gen)
    callback(c.paths.final)
    prune_empty_dirs(root=c.paths.final, ignore=['index.rst'])

    os.system(
        f'rm -rf {c.rst.dest}/{c.paths.final}; mv {c.paths.final} {c.rst.dest}'
    )


if __name__ == "__main__":
    main()
