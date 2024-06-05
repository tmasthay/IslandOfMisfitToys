import os
import re
import sys
from subprocess import check_output as co

import hydra
from omegaconf import DictConfig


def dir_up(path, n=1):
    for _ in range(n):
        path = os.path.dirname(path)
    return path


def idt_lines(s: str, *, idt_str='    ', idt_lvl=1):
    istr = idt_str * idt_lvl
    return istr + ('\n' + istr).join(s.split('\n'))


def sco(cmd, verbose=True):
    cmd = ' '.join(cmd.split())
    if verbose:
        print(cmd, flush=True)
    return co(cmd, shell=True).decode().strip()


def centralize_info(c: DictConfig):
    p = c.paths

    def get_paths(root):
        lines = sco(f"""
                find {root} -name "{c.param}_compare.yaml"
                    -exec grep -H "{c.score}" {{}} \; |
                awk -F':' '{{print $3,$1}}' |
                head -n {c.leaderboard_size} |
                sort -k1,1n
                """).strip().split('\n')
        lines = [e.strip() for e in lines]
        lines = [e.split() for e in lines if e]
        d = [
            {
                'score': e[0],
                'path': e[1],
                'target_path': e[1].replace(root, p.final),
            }
            for e in lines
        ]
        return d

    os.system(f'git clone --branch {c.git.branch} --single-branch {c.git.url}')
    dirs = get_paths(p.src)
    dirs.extend(get_paths(p.prev_leaders))

    dirs.sort(key=lambda x: (x["target_path"], float(x["score"])), reverse=True)

    final_size = min(c.leaderboard_size, len(dirs))
    dirs = dirs[:final_size]

    for d in dirs:
        lcl_dir = dir_up(d['target_path'], 3)
        repo_dir = dir_up(d['path'], 2)
        os.makedirs(lcl_dir, exist_ok=True)
        if os.path.exists(d['target_path']):
            raise ValueError(
                f'File {d["target_path"]} already exists...clear {p.final} and'
                ' re-run'
            )
        os.system(f'cp -r {repo_dir} {lcl_dir}')
    print(f'Written {final_size} files to {p.final}')


def extract_info(c: DictConfig):
    lines = sco(
        f'find {c.paths.final} -name "{c.param}_compare.yaml" -exec grep -H'
        f' "{c.score}" {{}} \; | awk -F\':\' \'{{print $3,$1}}\' | head -n'
        f' {c.leaderboard_size}'
    ).split('\n')
    lines = [e.strip().split() for e in lines]
    lines = [
        {
            'score': e[0],
            'path': e[1].replace(f"/meta/{c.param}_compare.yaml", ""),
            'images': [],
        }
        for e in lines
        if e
    ]
    for line in lines:
        path = line['path']
        line['img_files'] = []
        for ext in c.extensions:
            cmd = f'find {path} -name "*.{ext}"'
            line['img_files'].extend([e for e in sco(cmd).split('\n') if e])

        def sorter():
            d = c.rst.img.order.get(c.param, None)
            if d is None:
                return lambda v: 0
            else:

                def helper(v):
                    base = os.path.basename(v).split('.')[0]
                    if base in d:
                        return d.index(base)
                    else:
                        return float('inf')

                return helper

        line['img_files'].sort(key=sorter())

        # Read git and config files
        with open(os.path.join(path, 'git_info.txt'), 'r') as f:
            git_info = f.read().strip()
            sections = git_info.split(80 * '*')[:-1]
            line['git_info'] = {
                'short': sections[0].strip(),
                'diff': sections[1].strip(),
            }

        for hydra_file in ['config', 'overrides', 'hydra']:
            with open(
                os.path.join(path, '.hydra', f'{hydra_file}.yaml'), 'r'
            ) as f:
                line[hydra_file] = f.read().strip()

    lines.sort(key=lambda x: float(x['score']), reverse=False)
    return lines


def make_index_rst(leaderboard_dir, param):
    index_path = os.path.join(leaderboard_dir, 'index.rst')
    directories = [
        d
        for d in os.listdir(leaderboard_dir)
        if os.path.isdir(os.path.join(leaderboard_dir, d))
    ]

    with open(index_path, 'w') as file:
        s = """
Leaderboard
===========

.. toctree::
    :maxdepth: 1

""".lstrip()

        for dir in sorted(directories, key=lambda x: int(x)):
            s += f"    {dir}/{param}\n"
        file.write(s)


def setup_folders(*, name, size):
    rst_root = 'leaderboard'
    os.system(f'rm -rf {rst_root}')
    os.makedirs(rst_root, exist_ok=False)
    for i in range(size):
        curr_root = os.path.join(rst_root, f"{i+1}")
        os.makedirs(curr_root, exist_ok=False)
        os.makedirs(os.path.join(curr_root, 'figs'), exist_ok=False)


def write_param_file(*, c, rank, line):
    curr_root = os.path.join(c.folder_name, f"{rank+1}")
    rst_path = os.path.join(curr_root, f"{c.param}.rst")
    with open(rst_path, 'w') as rst_file:
        # rst_file.write(f"Score: {line['score']}\n")
        # rst_file.write(f"Run Path: {line['path']}\n")
        # rst_file.write(f"{line['git_info']['short']}\n")
        # rst_file.write("`Full Git Diff <full_git_diff>`_\n")
        # rst_file.write(f"overrides.yaml\n{line['overrides']}\n")
        # rst_file.write("`Config YAML <config>`_\n")
        # rst_file.write("`Hydra YAML <hydra>`_\n")
        # write hidden toctree so that sphinx will include dependent rst files
        img_str = ''
        for img in line['img_files']:
            os.system(f"cp {img} {curr_root}/figs/")
            img_str += (
                f".. image:: figs/{os.path.basename(img)}\n   :align:"
                " center\n\n"
            )
        title = f"Rank {rank+1}: {line['score']}"
        s = f"""
Rank {rank+1}: {line['score']}
{'=' * len(title)}

.. code-block::

    Score: {line['score']}
    Run Path: {line['path']}
    Git Info Summary:
    {idt_lines(line['git_info']['short'])}

.. toctree::
    :maxdepth: 1

    overrides
    config
    hydra
    full_git_diff

Images

{img_str}
""".lstrip()
        rst_file.write(s)


def write_git_diff_file(*, c, rank, line):
    curr_root = os.path.join(c.folder_name, f"{rank+1}")
    with open(os.path.join(curr_root, 'full_git_diff.rst'), 'w') as diff_file:
        title = 'Full Git Diff'
        diff_file.write(f"{title}\n{'=' * len(title)}\n\n")
        diff_file.write('.. code-block:: \n\n')
        diff_file.write(idt_lines(line['git_info']['diff']))


def write_yaml_rst_block_file(*, c, rank, line, filename):
    curr_root = os.path.join(c.folder_name, f"{rank+1}")
    with open(os.path.join(curr_root, f'{filename}.rst'), 'w') as curr_file:
        s = idt_lines(line[filename])
        rst_heading = (
            f"{filename.capitalize().replace('_', ' ')}\n"
            f"{'=' * len(filename)}\n\n"
            ".. code-block:: yaml\n\n"
        )
        curr_file.write(rst_heading + s)


@hydra.main(config_path="cfg", config_name="cfg", version_base=None)
def main(c: DictConfig):
    centralize_info(c)
    lines = extract_info(c)
    setup_folders(name=c.folder_name, size=len(lines))

    for rank, line in enumerate(lines):
        write_param_file(c=c, rank=rank, line=line)
        write_git_diff_file(c=c, rank=rank, line=line)
        for filename in ['config', 'hydra', 'overrides']:
            write_yaml_rst_block_file(
                c=c, rank=rank, line=line, filename=filename
            )

    # print(lines)
    os.makedirs(c.rst.dest, exist_ok=True)
    make_index_rst(c.rst.dest, param=c.param)
    os.system(
        f'rm -rf {c.rst.dest}/{c.folder_name}; mv {c.folder_name} {c.rst.dest}'
    )


if __name__ == "__main__":
    main()
