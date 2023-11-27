import os
import sys
from rich.table import Table
from rich.console import Console
import argparse
import torch
import numpy as np
from functools import wraps
from rich_tools import table_to_dicts


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
    curr_dir = os.path.dirname(__file__)
    alan_out_dir = os.path.join(curr_dir, 'alan', 'out', 'parallel')
    iomt_out_dir = os.path.join(curr_dir, 'val', 'out', 'parallel')
    filenames = ['vp', 'loss', 'out', 'out_filt']
    alan_files = extend_files(alan_out_dir, filenames)
    iomt_files = extend_files(iomt_out_dir, filenames)
    if not all_exist(alan_files.values()):
        print('Rerunning alan fwi_parallel.py')
        os.system('sleep 5')
        os.chdir(os.path.join(curr_dir, 'alan'))
        os.system('python fwi_parallel.py')
        os.chdir(curr_dir)
    if not all_exist(iomt_files.values()):
        print('Rerunning iomt fwi_parallel.py')
        os.system('sleep 5')
        os.chdir(os.path.join(curr_dir, 'val'))
        os.system('python fwi_parallel.py')
        os.chdir(curr_dir)
    if not all_exist(alan_files.values()) or not all_exist(iomt_files.values()):
        print('Error: could not generate all files')
        sys.exit(1)
    return alan_files, iomt_files


def get_tensors(filenames):
    return {k: torch.load(v) for k, v in filenames.items()}


def get_output():
    alan_files, iomt_files = get_files()
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


def clean_output(clean):
    alan_files, iomt_files = get_files()
    if 'a' in clean:
        for v in alan_files.values():
            os.remove(v)
    if 'i' in clean:
        for v in iomt_files.values():
            os.remove(v)


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
    clean_output(clean=args.clean)

    @transpose
    def rel_pointwise_error(alan_tensors, iomt_tensors):
        res = []
        for k in alan_tensors.keys():
            curr = []
            for u, v in zip(alan_tensors[k], iomt_tensors[k]):
                curr.append(f'{torch.norm(u - v) / torch.norm(u): .4e}')
            res.append(curr)
        return res

    return compare_output(
        row_gen=rel_pointwise_error,
        row_gen_label='Relative Pointwise Difference',
        output_filename=args.output,
        justify=args.justify,
    )


if __name__ == "__main__":
    args = get_args()
    main(args)
