import argparse
import os
import sys

from masthay_helpers.global_helpers import torch_dir_compare

from misfit_toys.utils import parse_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dir1', type=str)
    parser.add_argument('dir2', type=str)
    parser.add_argument('out', type=str)
    parser.add_argument('-f', '--figs', action='store_true')
    parser.add_argument(
        '-i', '--include-only', type=str, default=None, nargs='+'
    )
    args = parser.parse_args()

    args.dir1 = os.path.abspath(parse_path(args.dir1))
    args.dir2 = os.path.abspath(parse_path(args.dir2))
    args.out = os.path.abspath(parse_path(args.out))

    if not all([os.path.exists(e) for e in [args.dir1, args.dir2]]):
        raise ValueError(
            f'One of the directories does not exist, dir1={args.dir1},'
            f' dir2={args.dir2}'
        )

    os.makedirs(args.out, exist_ok=True)
    torch_dir_compare(
        args.dir1,
        args.dir2,
        args.out,
        include_only=args.include_only,
        make_figs=args.figs,
    )


if __name__ == '__main__':
    main()
