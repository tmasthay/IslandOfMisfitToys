import os
from subprocess import check_output as co

import hydra
from omegaconf import DictConfig


def make_index_rst(leaderboard_dir, param):
    index_path = os.path.join(leaderboard_dir, 'index.rst')
    directories = [
        d
        for d in os.listdir(leaderboard_dir)
        if os.path.isdir(os.path.join(leaderboard_dir, d))
    ]

    with open(index_path, 'w') as file:
        file.write("Leaderboard\n===========\n\n")
        file.write(
            "Welcome to the Leaderboard. Below are the links to the ranking"
            " details for each entry.\n\n"
        )

        for dir in sorted(directories, key=lambda x: int(x)):
            ranking_path = os.path.join(dir, f'{param}.rst')
            file.write(f"* `{dir} <{ranking_path}>`_\n")


def sco(cmd):
    return co(cmd, shell=True).decode().strip()


@hydra.main(config_path="cfg", config_name="cfg", version_base=None)
def main(c: DictConfig):
    lines = sco(
        f'find {c.root} -name "{c.param}_compare.yaml" -exec grep -H'
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

    rst_root = 'leaderboard'
    os.system(f'rm -rf {rst_root}')
    os.makedirs(rst_root, exist_ok=False)
    for rank, line in enumerate(lines):
        curr_root = os.path.join(rst_root, f"{rank+1}")
        os.makedirs(curr_root, exist_ok=False)
        os.makedirs(os.path.join(curr_root, 'figs'), exist_ok=False)
        rst_path = os.path.join(curr_root, f"{c.param}.rst")
        with open(rst_path, 'w') as rst_file:
            rst_file.write(f"Score: {line['score']}\n")
            rst_file.write(f"Run Path: {line['path']}\n")
            rst_file.write(f"{line['git_info']['short']}\n")
            rst_file.write("`Full Git Diff <full_git_diff>`_\n")
            rst_file.write(f"overrides.yaml\n{line['overrides']}\n")
            rst_file.write("`Config YAML <config>`_\n")
            rst_file.write("`Hydra YAML <hydra>`_\n")

            # Write images
            rst_file.write("\nImages\n======\n")
            for img in line['img_files']:
                # copy the file to local figs directory
                os.system(f"cp {img} {curr_root}/figs/")
                rst_file.write(
                    f".. image:: figs/{os.path.basename(img)}\n   :align:"
                    " center\n\n"
                )

            # write hidden toctree so that sphinx will include dependent rst files
            rst_file.write("\n.. toctree::\n   :hidden:\n\n")
            for filename in ['full_git_diff', 'config', 'hydra']:
                rst_file.write(f"   {filename}\n")

        # Write Git diff in a separate file
        with open(os.path.join(path, 'full_git_diff.rst'), 'w') as diff_file:
            diff_file.write(line['git_info']['diff'])
        for filename in ['config', 'hydra']:
            with open(
                os.path.join(curr_root, f'{filename}.rst'), 'w'
            ) as curr_file:
                s = '    ' + '\n    '.join(line[filename].split('\n'))
                rst_heading = '.. code-block:: yaml\n\n'
                curr_file.write(rst_heading + s)
    # print(lines)
    make_index_rst(rst_root, param=c.param)
    os.system(f'rm -rf {c.rst.dest}/{rst_root}; mv {rst_root} {c.rst.dest}')


if __name__ == "__main__":
    main()
