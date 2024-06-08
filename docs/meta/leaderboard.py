import os
import re
import sys
from os.path import join as pjoin
from subprocess import check_output as co

import hydra
import yaml
from mh.core import convert_dictconfig
from omegaconf import DictConfig


def dir_up(path, n=1):
    for _ in range(n):
        path = os.path.dirname(path)
    return path


def idt_lines(s: str, *, idt_str='    ', idt_lvl=1):
    istr = idt_str * idt_lvl
    return istr + ('\n' + istr).join(s.split('\n'))


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


def make_data_page(path: str, *, img_order, final_path) -> None:
    trunc_path = path.replace(final_path, '')
    if trunc_path.startswith('/'):
        trunc_path = trunc_path[1:]
    heading = trunc_path.split('/')[0]
    img_order = img_order[heading]
    if path.split('/')[-1] == 'figs':
        print(f"make_default_page: SKIP {path}")
        return
    # Ensure the path is a directory
    if not os.path.isdir(path):
        raise ValueError(f"The path '{path}' is not a directory.")

    # Build the content for index.rst
    content = []
    title = path.split('/')[-1]
    content.append(title)
    content.append("=" * len(title))
    content.append("")

    subdirs = [d for d in os.listdir(path) if os.path.isdir(pjoin(path, d))]
    files = [f for f in os.listdir(path) if os.path.isfile(pjoin(path, f))]

    subdirs = [e for e in subdirs if e not in ['figs']]

    # at first pass, we should have consolidated
    #    all files into "figs" and "metadata" subdirectories.
    #    Otherwise, something went wrong.
    if len(subdirs) != 1:
        raise ValueError(
            f"Expected 1 subdir, found {len(subdirs)=}, {subdirs=}"
        )
    if len(files) > 0:
        raise ValueError(f"Expected no files, found {len(files)=}, {files=}")

    if len(subdirs) + len(files) > 0:
        # Add the toctree for subdirectories
        content.append(".. toctree::")
        content.append("   :maxdepth: 1")
        content.append("   :caption: Contents:")
        content.append("")

        for subdir in subdirs:
            content.append(f"   {subdir}/index")

        content.append("")

        # Add collapsible buttons for each file in the directory
        for file in files:
            file_path = pjoin(path, file)
            with open(file_path, 'r') as file_content:
                file_data = file_content.read()

            file_data_idt = idt_lines(file_data, idt_lvl=2, idt_str='  ')

            file_ext = file.split('.')[-1]
            if file_ext not in ['yaml', 'py']:
                file_ext = 'text'
            content.extend(
                [
                    f".. admonition:: {file}",
                    f"   :class: toggle",
                    "",
                    f"   .. code-block:: {file_ext}",
                    "",
                    file_data_idt,
                ]
            )
    else:
        content.append("No content found.")
        content.append("")

    # Now process the images
    if os.path.exists(pjoin(path, 'figs')):
        figs = os.listdir(pjoin(path, 'figs'))
        figs = [e for e in figs if not e.startswith('.')]

        priority_dict = {e: i for i, e in enumerate(img_order)}
        figs = sorted(
            figs,
            key=lambda x: (priority_dict.get(x, float('inf')), x),
            reverse=True,
        )

        for fig in figs:
            content.extend(
                [
                    f".. image:: figs/{fig}",
                    "   :align: center",
                    "",
                ]
            )
    # Write the content to index.rst
    index_rst_path = os.path.join(path, 'index.rst')
    with open(index_rst_path, 'w') as f:
        f.write("\n".join(content))

    # print(f"index.rst generated at: {index_rst_path}")
    print(f"make_default_page: {path}")


def make_default_page(path: str) -> None:
    if path.split('/')[-1] == 'figs':
        print(f"make_default_page: SKIP {path}")
        return
    # Ensure the path is a directory
    if not os.path.isdir(path):
        raise ValueError(f"The path '{path}' is not a directory.")

    # Build the content for index.rst
    content = []
    title = path.split('/')[-1]
    content.append(title)
    content.append("=" * len(title))
    content.append("")

    subdirs = [
        d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))
    ]
    files = [
        f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))
    ]

    if len(subdirs) + len(files) > 0:
        # Add the toctree for subdirectories
        content.append(".. toctree::")
        content.append("   :maxdepth: 1")
        content.append("   :caption: Contents:")
        content.append("")

        numeric_subdirs = sorted(
            [e for e in subdirs if re.match(r'\d+', e)], key=lambda x: int(x)
        )
        nonnumeric_subdirs = sorted(
            [e for e in subdirs if not re.match(r'\d+', e)], key=lambda x: x
        )
        subdirs = nonnumeric_subdirs + numeric_subdirs
        for subdir in subdirs:
            content.append(f"   {subdir}/index")

        content.append("")

        # Add collapsible buttons for each file in the directory
        for file in files:
            file_path = os.path.join(path, file)
            with open(file_path, 'r') as file_content:
                file_data = file_content.read()

            file_data_idt = idt_lines(file_data, idt_lvl=2, idt_str='  ')

            file_ext = file.split('.')[-1]
            if file_ext not in ['yaml', 'py']:
                file_ext = 'text'
            content.extend(
                [
                    f".. admonition:: {file}",
                    f"   :class: toggle",
                    "",
                    f"   .. code-block:: {file_ext}",
                    "",
                    file_data_idt,
                ]
            )
    else:
        content.append("No content found.")
        content.append("")

    # Write the content to index.rst
    index_rst_path = os.path.join(path, 'index.rst')
    with open(index_rst_path, 'w') as f:
        f.write("\n".join(content))

    # print(f"index.rst generated at: {index_rst_path}")
    print(f"make_default_page: {path}")


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
    def get_paths(root):
        nonlocal reg_dict
        lines = sco(f"""
            find {root} -name "{param}_compare.yaml" |
            rev |
            cut -d'/' -f3- |
            rev |
            awk '{{print $0 "/.hydra/config.yaml"}}'
            """).split('\n')
        lines = [e for e in lines if e]
        # input('\n'.join(lines))
        for line in lines:
            og_path = line.replace('/.hydra/config.yaml', '')
            timestamp = ' '.join(
                [e for e in og_path.split('/')[-3:] if '-' in e]
            )

            # load the yaml
            cfg = yaml.load(open(line, 'r'), Loader=yaml.FullLoader)
            score_yaml = yaml.load(
                open(pjoin(og_path, f"meta/{param}_compare.yaml"), 'r'),
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
                # input(curr_dump_path)
                os.system(f"cp -r {e['og_path']} {curr_dump_path}")
                with open(
                    pjoin(curr_dump_path, f'{paths.meta}.yaml'), 'w'
                ) as f:
                    yaml.dump(e, f)
                os.system(f'mkdir -p {curr_dump_path}/metadata')
                os.system(
                    f'find {curr_dump_path} -type f ! -wholename "*figs*" -exec'
                    f' mv {{}} {curr_dump_path}/metadata \; 2>/dev/null'
                )
                os.system(
                    f'find {curr_dump_path} -mindepth 1 -type d -type d !'
                    ' -wholename "*figs*" ! -wholename "*metadata*" -exec rm'
                    ' -rf {} \; 2> /dev/null'
                )

        #     page_generator = get_callback(path=root_dump_path, idx_gen=idx_gen)
        #     page_generator(root_dump_path)

        # print(reg_dict)
        # sys.exit(1)

        # return lines

    get_paths(paths.src)
    get_paths(paths.prev_leaders)
    deploy()

    # all_dirs = bottom_up_dirs(paths.final)
    for dir in bottom_up_dirs(pjoin(paths.final, param)):
        callback = get_callback(path=dir, idx_gen=idx_gen)
        callback(dir)

    # print(init_dirs)
    # sys.exit(1)


def extract_info(
    *,
    path: str,
    param: str,
    score: str,
    leaderboard_size: int,
    order: list,
    extensions: list,
):
    lines = sco(
        f'find {path} -name "{param}_compare.yaml" -exec grep -H'
        f' "{score}" {{}} \; | awk -F\':\' \'{{print $3,$1}}\' | head -n'
        f' {leaderboard_size}'
    ).split('\n')
    lines = [e.strip().split() for e in lines]
    lines = [
        {
            'score': e[0],
            'path': e[1].replace(f"/meta/{param}_compare.yaml", ""),
            'images': [],
        }
        for e in lines
        if e
    ]
    for line in lines:
        path = line['path']
        line['img_files'] = []
        cfg = yaml.load(
            open(pjoin(path, '.hydra', 'config.yaml'), 'r'),
            Loader=yaml.FullLoader,
        )
        line['target_path'] = cfg['case']['data']['path'].split('data/')[-1]
        # input(line)
        for ext in extensions:
            cmd = f'find {path} -name "*.{ext}"'
            line['img_files'].extend([e for e in sco(cmd).split('\n') if e])

        def sorter():
            d = order.get(param, None)
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
        with open(pjoin(path, 'git_info.txt'), 'r') as f:
            git_info = f.read().strip()
            sections = git_info.split(80 * '*')[:-1]
            line['git_info'] = {
                'short': sections[0].strip(),
                'diff': sections[1].strip(),
            }

        for hydra_file in ['config', 'overrides', 'hydra']:
            with open(pjoin(path, '.hydra', f'{hydra_file}.yaml'), 'r') as f:
                line[hydra_file] = f.read().strip()

    lines.sort(key=lambda x: float(x['score']), reverse=False)
    return lines


def make_param_rst(leaderboard_dir, param):
    index_path = pjoin(leaderboard_dir, 'index.rst')
    directories = [
        d
        for d in os.listdir(leaderboard_dir)
        if os.path.isdir(pjoin(leaderboard_dir, d))
    ]

    with open(index_path, 'w') as file:
        s = f"""
{param}
===========

.. toctree::
    :maxdepth: 1

""".lstrip()

        for dir in sorted(directories, key=lambda x: int(x)):
            s += f"    {dir}/{param}\n"
        file.write(s)


def make_index_rst(*, root, params):
    index_path = pjoin(root, 'index.rst')

    with open(index_path, 'w') as file:
        s = """
Leaderboard
===========

.. toctree::
    :maxdepth: 1

""".lstrip()

        for param in params:
            s += f"    {param}/index\n"
        file.write(s)


def setup_folders(*, name, size):
    os.system(f'rm -rf {name}')
    os.makedirs(name, exist_ok=False)
    for i in range(size):
        curr_root = pjoin(name, f"{i+1}")
        os.makedirs(curr_root, exist_ok=False)
        os.makedirs(pjoin(curr_root, 'figs'), exist_ok=False)


def write_param_file(*, folder_name, param, line):
    folder_name = pjoin(folder_name, line['target_path'])
    rank = 1 + int(sco(f'ls {folder_name} | grep "[0-9][0-9]*" | wc -l'))
    curr_root = pjoin(folder_name, f"{rank}")
    rst_path = pjoin(curr_root, "index.rst")

    os.makedirs(curr_root, exist_ok=False)
    os.makedirs(pjoin(curr_root, 'figs'), exist_ok=False)
    # input(f"{curr_root=}, {rst_path=}")
    with open(rst_path, 'w') as rst_file:
        img_str = ''
        for img in line['img_files']:
            os.system(f"cp {img} {curr_root}/figs/")
            img_str += (
                f".. image:: figs/{os.path.basename(img)}\n   :align:"
                " center\n\n"
            )
        title = f"Rank {rank}: {line['score']}"
        s = f"""
Rank {rank}: {line['score']}
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
    return curr_root


def write_git_diff_file(*, path, line):
    with open(pjoin(path, 'full_git_diff.rst'), 'w') as diff_file:
        title = 'Full Git Diff'
        diff_file.write(f"{title}\n{'=' * len(title)}\n\n")
        diff_file.write('.. code-block:: \n\n')
        diff_file.write(idt_lines(line['git_info']['diff']))


def write_yaml_rst_block_file(*, path, line, filename):
    # curr_root = pjoin(path, f"{rank+1}")
    curr_root = path
    with open(pjoin(curr_root, f'{filename}.rst'), 'w') as curr_file:
        s = idt_lines(line[filename])
        rst_heading = (
            f"{filename.capitalize().replace('_', ' ')}\n"
            f"{'=' * len(filename)}\n\n"
            ".. code-block:: yaml\n\n"
        )
        curr_file.write(rst_heading + s)


def write_folder_structure(*, search_root, out_root, params):
    def make_folders(param, verbose=True):
        res_cmd = sco(f"""
            find {search_root} -type d |
            grep -v "__pycache__" |
            sed -E 's|{search_root}/||' |
            tail -n +2 |
            awk '{{print "mkdir -p {pjoin(out_root, param)}{os.sep}" $0 ";"}}'
        """)
        return os.system(res_cmd)

    for param in params:
        make_folders(param)

    directories = sco(
        f"""find {out_root} -mindepth 1 -type d -exec sh -c 'echo $1:$(ls "$1")' _ {{}} \;"""
    ).split('\n')
    directories = [[ee.strip() for ee in e.split(':')] for e in directories]

    for dir, files in directories:
        # name = dir.split(os.sep)[-1]
        files = [f"{e}/index" for e in files.split()]
        # toc = "\n    ".join(files)
        if len(files) == 0:
            continue


#         with open(f'{dir}/index.rst', 'w') as f:
#             s = f"""
# {name}
# {'=' * len(name)}
# .. toctree::
#     :maxdepth: 1

#     {toc}
# """.lstrip()
#             f.write(s)


def toctree(*, title=None, files, maxdepth=1):
    # files = [f"{f}/index" for f in files]
    s = "" if title is None else f"{title}\n{'=' * len(title)}\n\n"
    s += f".. toctree::\n    :maxdepth: {maxdepth}\n\n"
    for filename in files:
        if filename in ['\n', ''] or '.. _spacer1:' in filename:
            s += filename
        else:
            s += f"    {filename}/index\n"
    # s += "\n    " + "\n    ".join(files)
    s += "\n\n"
    return s


def write_dynamic_index_rst(*, root):
    filenames = os.listdir(root)

    if 'index.rst' in filenames:
        return

    numbered_files = [e for e in filenames if re.match(r'\d+', e)]
    nonnumbered_files = [e for e in filenames if not re.match(r'\d+', e)]

    if len(numbered_files) > 0:
        numbered_files.sort(key=lambda x: int(x))
        nonnumbered_files.extend(['\n', '    .. _spacer1:\n', '\n'])
        nonnumbered_files.extend(numbered_files)

    name = root.split(os.sep)[-1]
    s = f"{name}\n{'=' * len(name)}\n\n"
    with open(pjoin(root, 'index.rst'), 'w') as f:
        f.write(s)
        if len(nonnumbered_files) > 0:
            # nonnumbered_toctree = toctree(title='Leaderboard (and derived datasets on top)', files=nonnumbered_files)
            nonnumbered_toctree = toctree(title='', files=nonnumbered_files)
            f.write(nonnumbered_toctree)
        # if len(numbered_files) > 0:
        #     numbered_toctree = toctree(title='Leaderboard', files=numbered_files)
        #     f.write(numbered_toctree)


def write_all_dynamic_index_rst(*, root):
    for dirpath, dirnames, filenames in os.walk(root):
        write_dynamic_index_rst(root=dirpath)


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
        # curr_fold = pjoin(c.folder_name, param)
        # input(f'{param=}')
        centralize_info(
            paths=c.paths,
            param=param,
            score=c.score,
            leaderboard_size=c.leaderboard_size,
            idx_gen=c.rst.idx_gen,
        )
    #     lines = extract_info(
    #         path=c.paths.final,
    #         param=param,
    #         score=c.score,
    #         leaderboard_size=c.leaderboard_size,
    #         order=c.rst.img.order,
    #         extensions=c.extensions,
    #     )
    #     # input(lines)
    #     # setup_folders(name=curr_fold, size=len(lines))

    #     for line in lines:
    #         root = write_param_file(
    #             folder_name=curr_fold, param=param, line=line
    #         )
    #         write_git_diff_file(path=root, line=line)
    #         for filename in ['config', 'hydra', 'overrides']:
    #             write_yaml_rst_block_file(
    #                 path=root, line=line, filename=filename
    #             )

    #     # # print(lines)
    #     # make_param_rst(curr_fold, param=param)

    # # make_index_rst(root=c.folder_name, params=c.params)
    # write_all_dynamic_index_rst(root=c.folder_name)
    # os.system(
    #     f'rm -rf {c.rst.dest}/{c.folder_name}; mv {c.folder_name} {c.rst.dest}'
    # )
    # os.system(
    #     f'rm -rf {c.rst.dest}/{c.paths.final}; mv {c.paths.final} {c.rst.dest}'
    # )
    callback = get_callback(path=c.paths.final, idx_gen=c.rst.idx_gen)
    callback(c.paths.final)

    os.system(
        f'rm -rf {c.rst.dest}/{c.paths.final}; mv {c.paths.final} {c.rst.dest}'
    )


if __name__ == "__main__":
    main()
