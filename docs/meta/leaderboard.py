import os
from subprocess import check_output as co

import hydra
from omegaconf import DictConfig


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
        for ext in c.extensions:
            line['images'].extend(
                [
                    e
                    for e in sco(f'find {path} -name "*.{ext}"').split('\n')
                    if e
                ]
            )
        with open(os.path.join(path, 'git_info.txt'), 'r') as f:
            git_info = f.read().strip()
            sections = git_info.split(80 * '*')[:-1]
            line['git_info'] = {
                'short': sections[0].strip(),
                'diff': sections[1].strip(),
            }

        for hydra_files in ['config', 'overrides', 'hydra']:
            with open(
                os.path.join(path, '.hydra', f'{hydra_files}.yaml'), 'r'
            ) as f:
                line[hydra_files] = f.read().strip()
    lines.sort(key=lambda x: float(x['score']), reverse=False)
    print(lines)


if __name__ == "__main__":
    main()
