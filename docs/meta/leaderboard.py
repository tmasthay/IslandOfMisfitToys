import datetime
import os
import re
import sys
from os.path import join as pjoin
from subprocess import check_output as co

import hydra
import yaml
from mh.core import convert_dictconfig
from omegaconf import DictConfig


def idt_lines(s: str, *, idt_str='    ', idt_lvl=1):
    istr = idt_str * idt_lvl
    return istr + ('\n' + istr).join(s.split('\n'))


def categorize_files(files, groups):
    u = {k: [] for k in groups}
    u['other'] = []
    for f in files:
        found_regex = False
        for group, regexs in groups.items():
            for regex in regexs:
                if re.match(regex, f):
                    u[group].append(f)
                    found_regex = True
                    break
        if not found_regex:
            u['other'].append(f)

    for k, v in u.items():
        u[k] = sorted(v)
    return u


def group_admonitions(d, idt_char='  ', path='') -> str:
    # idt = lambda x, y: idt_char * x + y
    def idt(x, y):
        return idt_char * x + y

    def lcl_idt_lines(s: str, *, idt_lvl):
        return idt_lines(s, idt_str=idt_char, idt_lvl=idt_lvl)

    def make_admonition(*, idt_level, heading) -> str:
        s = idt(idt_level, f'.. admonition:: {heading}\n')
        s += idt(idt_level + 1, ':class: toggle\n\n')
        return s.split('\n')

    def make_code_block(*, idt_level, filename):
        file_type = filename.split('.')[-1]
        if file_type not in ['yaml', 'py', 'c', 'cpp', 'h', 'hpp']:
            file_type = 'text'
        s = idt(idt_level, f'.. code-block:: {file_type}\n\n')
        file_content = open(pjoin(path, filename), 'r').read()
        file_content = file_content.strip() if file_content else 'Empty file'
        file_content = lcl_idt_lines(file_content, idt_lvl=idt_level + 1)
        s += f'{file_content}\n\n'
        return s.split('\n')

    # a = [
    #     'Title',
    #     '=====\n',
    # ]
    a = []
    metadata_section_heading = 'Metadata'
    a.extend(
        f'{metadata_section_heading}\n{"-" * len(metadata_section_heading)}\n\n'
        .split('\n')
    )
    a.extend(make_admonition(idt_level=0, heading='Metadata'))
    for k, v in d.items():
        a.extend(make_admonition(idt_level=1, heading=k))
        for e in v:
            file_type = e.split('.')[-1]
            if file_type not in ['yaml', 'py']:
                file_type = 'text'
            a.extend(make_admonition(idt_level=2, heading=e))
            a.extend(make_code_block(idt_level=3, filename=e))
    return '\n'.join(a)


def make_data_page(
    path: str, *, img_order, final_path, img_first, groups, maxdepth
) -> None:
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
    toc_content = []
    img_content = []
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
    if len(subdirs) != 0:
        raise ValueError(
            f"Expected 0 subdir, found {len(subdirs)=}, {subdirs=}"
        )
    if len(files) == 0:
        raise ValueError(
            f"Expected nonzero number of files, found {len(files)=}, {files=}"
        )
    if 'config.yaml' not in files:
        raise ValueError(
            f"Expected 'config.yaml' in files, found {files=}, {path=}"
        )

    if len(subdirs) + len(files) > 0:
        # Add the toctree for subdirectories
        toc_content.append(".. toctree::")
        toc_content.append(f"   :maxdepth: {maxdepth}")
        toc_content.append("   :caption: Contents:")
        toc_content.append("")

        for subdir in subdirs:
            toc_content.append(f"   {subdir}/index")

        toc_content.append("")

        # Add collapsible buttons for each file in the directory

        # First add main admonition
        # toc_content.extend([".. admonition:: Metadata", "  :class: toggle", ""])
        # for file in files:
        #     file_path = pjoin(path, file)
        #     with open(file_path, 'r') as file_content:
        #         file_data = file_content.read()

        #     file_data_idt = idt_lines(file_data, idt_lvl=3, idt_str='  ')

        #     file_ext = file.split('.')[-1]
        #     if file_ext not in ['yaml', 'py']:
        #         file_ext = 'text'
        #     toc_content.extend(
        #         [
        #             f"  .. admonition:: {file}",
        #             "    :class: toggle",
        #             "",
        #             f"    .. code-block:: {file_ext}",
        #             "",
        #             file_data_idt or "Empty file",
        #         ]
        #     )
        regroup_dict = categorize_files(files, groups)
        toc_content.append(group_admonitions(regroup_dict, path=path))
    else:
        toc_content.append("No content found.")
        toc_content.append("")

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
            heading = fig.split('.')[0]
            img_content.extend(
                [
                    heading,
                    "-" * len(heading),
                    "",
                    f".. image:: figs/{fig}",
                    "   :align: center",
                    "",
                ]
            )

    if img_first:
        content.extend(img_content + toc_content)
    else:
        content.extend(toc_content + img_content)

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
        # First, add a "main" button
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
                    "   :class: toggle",
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
    reg_dict['unregistered_tests'] = []

    # registered_tests = [e for e in registered_tests if e]
    def get_paths(root, *, cfg_name='meta'):
        nonlocal reg_dict
        lines = sco(f"""
            find {root} -name "{param}_compare.yaml" || true
            """).split('\n')
        lines = [e for e in lines if e]
        # input('\n'.join(lines))
        for line in lines:
            with open(line, 'r') as f:
                meta = yaml.load(f, Loader=yaml.FullLoader)
                if meta['proj_path'] in reg_dict:
                    reg_dict[meta['proj_path']].append(meta)
                else:
                    reg_dict['unregistered_tests'].append(meta)

    def deploy():
        nonlocal reg_dict
        for k, v in reg_dict.items():
            reg_dict[k] = sorted(v, key=lambda x: float(x[score]), reverse=True)

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
                os.system(f"cp -r {e['root']} {curr_dump_path}")
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

    get_paths(paths.src, cfg_name='resolved_config')
    get_paths(paths.prev_leaders, cfg_name=None)
    # input(reg_dict)
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
