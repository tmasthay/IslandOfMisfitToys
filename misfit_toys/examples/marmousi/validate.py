import argparse
import os
import sys
from functools import wraps

import matplotlib.pyplot as plt
import numpy as np
import torch
from mh.typlotlib_legacy import plot_tensor2d_fast
from returns.curry import curry
from rich.console import Console
from rich.table import Table
from rich_tools import table_to_dicts

from misfit_toys.examples.marmousi.alan.fwi_parallel import main as alan
from misfit_toys.examples.marmousi.val.fwi_parallel import main as iomt


def extend_files(path, filenames):
    return {
        filename: os.path.join(path, f'{filename}_record.pt')
        for filename in filenames
    }


def all_exist(filenames):
    return all([os.path.exists(filename) for filename in filenames])


def transpose(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        result = f(*args, **kwargs)
        return np.array(result).T.tolist()

    return wrapper


def get_files():
    ran_alan = False
    ran_iomt = False
    curr_dir = os.path.dirname(__file__)
    alan_out_dir = os.path.join(curr_dir, 'alan', 'out', 'parallel')
    iomt_out_dir = os.path.join(curr_dir, 'val', 'out', 'parallel')
    filenames = ['vp', 'loss', 'out', 'out_filt']
    alan_files = extend_files(alan_out_dir, filenames)
    iomt_files = extend_files(iomt_out_dir, filenames)
    if not all_exist(alan_files.values()):
        print("RUNNING ALAN'S VERSION")
        alan()
        ran_alan = True
        if not all_exist(alan_files.values()):
            root = os.environ.get('ISL', 'IOMT_ROOT')
            path = os.path.join(
                root,
                'misfit_toys',
                'examples',
                'marmousi',
                'alan',
                'fwi_parallel.py',
            )
            raise ValueError(
                "Error: Alan's version did not generate all files\n"
                f"See file {os.path.abspath(path)}"
            )
    if not all_exist(iomt_files.values()):
        print("RUNNING IOMT VERSION")
        iomt()
        ran_iomt = True
    if not all_exist(alan_files.values()) or not all_exist(iomt_files.values()):
        print('Error: could not generate all files')
        sys.exit(1)
    return alan_files, iomt_files, ran_alan, ran_iomt


def get_tensors(filenames):
    return {k: torch.load(v) for k, v in filenames.items()}


def get_output():
    alan_files, iomt_files, _, _ = get_files()
    return get_tensors(alan_files), get_tensors(iomt_files)


def compare_output(
    *, row_gen, row_gen_label, output_filename='validate.txt', justify='right'
):
    alan_tensors, iomt_tensors = get_output()
    table = Table(title=row_gen_label)
    for k in alan_tensors.keys():
        table.add_column(k, justify=justify)
    row_data = row_gen(alan_tensors, iomt_tensors)
    for row in row_data:
        table.add_row(*row)
    if output_filename is None:
        console = Console()
    else:
        console = Console(file=open(output_filename, 'w'))
    console.print(table)
    res = list(table_to_dicts(table))
    if len(res) > 0:
        res = {k: [float(d[k]) for d in res] for k in res[0].keys()}
    else:
        res = {}
    return res


def make_gifs(out_dir):
    alan_tensors, iomt_tensors = get_output()

    diff_tensors = {
        k: alan_tensors[k] - iomt_tensors[k] for k in alan_tensors.keys()
    }
    tensors = {'alan': alan_tensors, 'iomt': iomt_tensors, 'diff': diff_tensors}

    labels = {
        'vp': ['Extent', 'Depth', 'Epoch'],
        'out': ['Extent', 'Depth', 'Time'],
        'out_filt': ['Extent', 'Depth', 'Time'],
    }

    permutations = {
        'vp': (2, 1, 0),
        'out': (2, 3, 1, 0),
        'out_filt': (2, 3, 1, 0),
    }

    def get(x):
        return os.path.join(out_dir, 'figs', x)

    dirs = {'alan': get('alan'), 'iomt': get('iomt'), 'diff': get('diff')}

    for k, v in dirs.items():
        os.makedirs(v, exist_ok=True)

    @curry
    def config(title, *, labels):
        plt.title(title)
        plt.xlabel(labels[0])
        plt.ylabel(labels[1])
        plt.colorbar()

    for case, tensor_group in tensors.items():
        for k, label in labels.items():
            plot_tensor2d_fast(
                tensor=tensor_group[k].permute(*permutations[k]),
                labels=label,
                config=config(labels=label),
                cmap='seismic',
                name=k,
                path=dirs[case],
            )
        plt.clf()
        plt.plot(range(len(tensor_group['loss'])), tensor_group['loss'])
        plt.title(f'Loss {case}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig(os.path.join(dirs[case], 'loss.jpg'))


def clean_output(clean, out_file_name):
    alan_files, iomt_files, ran_alan, ran_iomt = get_files()
    if 'a' in clean and not ran_alan:
        for v in alan_files.values():
            os.remove(v)
    if 'i' in clean and not ran_iomt:
        for v in iomt_files.values():
            os.remove(v)
    if os.path.exists(out_file_name):
        os.remove(out_file_name)


def get_args():
    parser = argparse.ArgumentParser(
        description='Compare output of deepwave example vs IOMT'
    )
    parser.add_argument(
        '-o',
        '--output',
        type=str,
        default='validate.txt',
        help='Output filename',
    )
    parser.add_argument(
        '-j',
        '--justify',
        type=str,
        default='right',
        help='Justification for table',
    )
    parser.add_argument('-c', '--clean', type=str, default='')
    args = parser.parse_args()
    return args


def main(args):
    clean_output(clean=args.clean, out_file_name=args.output)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    @transpose
    def rel_pointwise_error(alan_tensors, iomt_tensors):
        res = []
        for k in alan_tensors.keys():
            curr = []
            for u, v in zip(alan_tensors[k], iomt_tensors[k]):
                curr.append(f'{torch.norm(u - v) / torch.norm(u): .4e}')
            res.append(curr)
        return res

    res = compare_output(
        row_gen=rel_pointwise_error,
        row_gen_label='Relative Pointwise Difference',
        output_filename=args.output,
        justify=args.justify,
    )

    do_plots = False
    if do_plots:
        make_gifs(os.path.dirname(args.output))
    return res


if __name__ == "__main__":
    args = get_args()
    main(args)
