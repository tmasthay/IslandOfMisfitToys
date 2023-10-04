from subprocess import check_output as co
from subprocess import CalledProcessError
import sys
from time import time
import matplotlib.pyplot as plt
from imageio import imread, mimsave
import numpy as np
import torch
from typing import Annotated as Ant, Any, Optional as Opt, Callable
from abc import ABCMeta, abstractmethod
import itertools
from .swiffer import *
from torch.optim.lr_scheduler import _LRScheduler
import deepwave as dw
from warnings import warn
import os
import textwrap


class DotDict:
    def __init__(self, d):
        for k, v in d.items():
            setattr(self, k, v)

    def set(self, k, v):
        setattr(self, k, v)

    def get(self, k):
        return getattr(self, k)

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def has(self, k):
        return hasattr(self, k)

    def has_all(self, *keys):
        return all([self.has(k) for k in keys])

    def has_all_type(self, *keys, lcl_type=None):
        return all(
            [self.has(k) and type(self.get(k)) is lcl_type for k in keys]
        )


def parse_path(path):
    if path is None:
        path = 'conda'
    if path.startswith('conda'):
        path = path.replace('conda', os.environ['CONDA_PREFIX'])
    elif path.startswith('pwd'):
        path = path.replace('pwd', os.getcwd())
    else:
        path = os.path.join(os.getcwd(), path)
    return path


def auto_path(kw_path='path', make_dir=False):
    def decorator(func):
        def wrapper(*args, **kwargs):
            if kw_path in kwargs:
                kwargs[kw_path] = parse_path(kwargs[kw_path])
                if make_dir:
                    os.makedirs(kwargs[kwargs[kw_path]], exist_ok=True)
            return func(*args, **kwargs)

        return wrapper

    return decorator


def get_pydict(path, *, filename='metadata', as_class=False):
    path = parse_path(path)
    filename = filename.replace('.pydict', '') + '.pydict'
    full_path = os.path.join(path, filename)
    d = eval(open(full_path, 'r').read())
    if as_class:
        return DotDict(d)
    else:
        return d


def gpu_mem(msg='', color='red', print_protocol=print):
    if len(msg) > 0 and msg[-1] != '\n':
        msg += '\n'

    if type(color) == tuple:
        color = [str(e) for e in color]
        color = 'rgb' + '_'.join(color)
    out = sco_bash('gpu_mem', color, split=True)
    out = [f'    {e}' for e in out if len(e) > 0]
    out[-1] = out[-1].replace('\n', '')
    out = '\n'.join(out)
    print_protocol(f'{msg}{out}')


def gaussian_perturb(ref, scaled_sigma, scaled_mu, scale=False):
    if scale:
        scaling = torch.max(torch.abs(ref))
    else:
        scaling = 1.0
    sigma = scaled_sigma * scaling
    mu = scaled_mu * scaling
    noise = torch.randn_like(ref) * sigma + mu
    tmp = ref + noise
    v = tmp.clone().requires_grad_()
    return v


def verbosity_str_to_int(*, verbosity, levels):
    if type(verbosity) == int:
        return verbosity
    elif type(verbosity) == str:
        verbosity = verbosity.lower()
        for level, level_names in levels:
            if verbosity in level_names:
                return level
        raise ValueError(f'Verbose value {verbosity} not recognized')
    else:
        raise ValueError(f'Verbosity must be int or str, got {type(verbosity)}')


def clean_levels(levels):
    if levels is None:
        levels = []
        levels.append((0, ['none', 'silent']))
        levels.append((1, ['low', 'progress']))
        levels.append((2, ['medium', 'debug']))
        levels.append((np.inf, ['high', 'all']))
    for i, l in enumerate(levels):
        if type(l) is int:
            levels[i] = (l, [str(l)])
    for i, l in enumerate(levels):
        if type(l) not in [list, tuple] or len(l) != 2:
            raise ValueError('Levels must be list of pairs')
        elif type(l[1]) is not list:
            raise ValueError(f'Level names must be list, got {type(l[1])}')
        elif str(l[0]) not in l[1]:
            l[1].append(str(l[0]))
            if l[0] is np.inf:
                l[1].append('infinity')
    levels = sorted(levels, key=lambda x: x[0])
    return levels


def run_verbosity(*, verbosity, levels):
    levels = clean_levels(levels)
    v2i = lambda x: verbosity_str_to_int(verbosity=x, levels=levels)
    verbosity_int = v2i(verbosity)

    def helper(f):
        def helper_inner(*args, _verbosity_, **kw):
            _verbosity_int = v2i(_verbosity_)
            if _verbosity_int <= verbosity_int:
                return f(*args, **kw)
            else:
                return None

        return helper_inner

    return helper


def mem_report(*args, precision=2, sep=', ', rep=None):
    filtered_args = []
    if rep is None:
        rep = []
    [rep.append('unknown') for _ in range(len(args) - len(rep))]
    add = lambda x, i: filtered_args.append(x + ' (' + rep[i] + ')')
    for i, arg in enumerate(args):
        if 1e18 < arg:
            add(f'{arg/1e18:.{precision}f} EB', i)
        elif 1e15 < arg:
            add(f'{arg/1e15:.{precision}f} PB', i)
        elif 1e12 < arg:
            add(f'{arg/1e12:.{precision}f} TB', i)
        elif 1e9 < arg:
            add(f'{arg/1e9:.{precision}f} GB', i)
        elif 1e6 < arg:
            add(f'{arg/1e6:.{precision}f} MB', i)
        elif 1e3 < arg:
            add(f'{arg/1e3:.{precision}f} KB', i)
        else:
            add(f'{arg:.{precision}f} B', i)
    return sep.join(filtered_args)


def full_mem_report(precision=2, sep=', ', rep=('free', 'total'), title=None):
    if title is None:
        title = ''
    else:
        title = title + '\n    '
    return title + mem_report(
        *torch.cuda.mem_get_info(), precision=precision, sep=sep, rep=rep
    )


def taper(x, length):
    return dw.common.cosine_taper_end(x, length)


def summarize_tensor(tensor, *, idt_level=0, idt_str='    ', heading='Tensor'):
    stats = dict(dtype=tensor.dtype, shape=tensor.shape)
    if tensor.dtype == torch.bool:
        return str(stats)
    elif tensor.dtype in [
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    ]:
        tensor = tensor.float()

    # Compute various statistics
    stats.update(
        dict(
            mean=torch.mean(tensor).item(),
            variance=torch.var(tensor).item(),
            median=torch.median(tensor).item(),
            min=torch.min(tensor).item(),
            max=torch.max(tensor).item(),
            stddev=torch.std(tensor).item(),
        )
    )

    # Prepare the summary string with the desired indentation
    indent = idt_str * idt_level
    summary = [f"{heading}:"]
    for key, value in stats.items():
        summary.append(f"{indent}{idt_str}{key} = {value}")

    return '\n'.join(summary)


def print_tensor(tensor, print_fn=print, print_kwargs=None, **kwargs):
    if print_kwargs is None:
        print_kwargs = {'flush': True}
    print_fn(summarize_tensor(tensor, **kwargs), **print_kwargs)


def downsample_any(u, ratios):
    assert len(ratios) == len(u.shape), (
        f'downsample_any: len(ratios)={len(ratios)} !='
        f' len(u.shape)={len(u.shape)}'
    )
    assert all(
        [r > 0 and type(r) is int for r in ratios]
    ), f'downsample_any: ratios={ratios} must be positive ints'

    slices = [slice(None, None, r) for r in ratios]
    return u[tuple(slices)]


class SlotMeta(type):
    def __new__(cls, name, bases, class_dict):
        # Extract the variable names from the annotations
        try:
            annotated_keys = list(class_dict['__annotations__'].keys())
        except KeyError:
            annotated_keys = []

        # Find attributes that are not methods, not in special names and not already annotated
        non_annotated_attrs = [
            key
            for key, value in class_dict.items()
            if not (
                callable(value) or key.startswith('__') or key in annotated_keys
            )
        ]

        # Add the default annotations for non-annotated attributes
        for key in non_annotated_attrs:
            class_dict['__annotations__'][key] = Ant[Any, 'NOT ANNOTATED']

            # Optional: Remove the attributes as they'll be defined by __slots__
            class_dict.pop(key, None)

        # Create the __slots__ attribute from updated annotationsi
        try:
            class_dict['__slots__'] = list(class_dict['__annotations__'].keys())
        except KeyError:
            class_dict['__slots__'] = []

        return super().__new__(cls, name, bases, class_dict)


class CombinedMeta(SlotMeta, ABCMeta):
    pass


def idt_print(*args, levels=None, idt='    '):
    if levels is None:
        levels = [1 for _ in range(len(args))]
        levels[0] = 0
    elif type(levels) is int:
        tmp = levels
        levels = [tmp + 1 for _ in range(len(args))]
        levels[0] = tmp

    lines = args
    i = 0
    for arg, idt_level in zip(args, levels):
        lines[i] = f'{idt * idt_level}{arg}'
    return '\n'.join(lines)
