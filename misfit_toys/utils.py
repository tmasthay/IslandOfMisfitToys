"""
Utility functions for the Island of Misfit Toys project.

Functions:
- setup: Set up the distributed training environment.
- cleanup: Clean up the distributed training environment.
- get_file: Get the file path for saving or loading a tensor.
- load: Load a tensor from a file.
- load_all: Load tensors from multiple files.
- save: Save a tensor to a file.
- savefig: Save the current figure to a file.
- filt: Apply a biquad filter to the input signal.
- parse_path: Parse the input path.
- auto_path: Decorator to automatically parse and create directories for file paths.
- get_pydict: Get a Python dictionary from a file.
- gaussian_perturb: Generate a Gaussian perturbation based on a reference tensor.
- verbosity_str_to_int: Convert a verbosity string to an integer value.
- clean_levels: Clean and validate the verbosity levels.
- run_verbosity: Decorator to control the verbosity of a function.
- mem_report: Generate a memory report.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import deepwave as dw
from typing import Annotated as Ant, Any
from masthay_helpers.global_helpers import find_files, vco, ctab, DotDict
import torch.distributed as dist
from torchaudio.functional import biquad

# Rest of the code...
import matplotlib.pyplot as plt
import numpy as np
import torch

from typing import Annotated as Ant, Any

import deepwave as dw
import os

from masthay_helpers.global_helpers import find_files, vco, ctab, DotDict
import torch.distributed as dist
from torchaudio.functional import biquad
import glob
import socket


def find_available_port(start_port, max_attempts=5):
    """
    Tries to find an available network port starting from 'start_port'.
    It makes up to 'max_attempts' to find an available port.
    Returns the first available port number or raises an exception if no available port is found.
    """
    port = start_port
    for _ in range(max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", port))
                return port  # Port is available
        except OSError as e:
            if e.errno == 98:  # Port is already in use
                print(f"Port {port} is in use, trying next port.")
                port += 1
            else:
                raise  # Re-raise exception if it's not a "port in use" error


def setup(rank, world_size, port=12355):
    """
    Set up the distributed training environment.

    Args:
        rank (int): The rank of the current process.
        world_size (int): The total number of processes.
        port (int, optional): The port number for communication. Defaults to 12355.
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    """
    Clean up the distributed training environment.
    """
    dist.destroy_process_group()


def get_file(name, *, rank="", path="out/parallel", ext=".pt"):
    """
    Get the file path for saving or loading a tensor.

    Args:
        name (str): The name of the file.
        rank (str, optional): The rank of the current process. Defaults to "".
        path (str, optional): The directory path. Defaults to "out/parallel".
        ext (str, optional): The file extension. Defaults to ".pt".

    Returns:
        str: The file path.
    """
    ext = "." + ext.replace(".", "")
    name = name.replace(ext, "")
    if rank != "":
        rank = f"_{rank}"
    return os.path.join(os.path.dirname(__file__), path, f"{name}{rank}{ext}")


def load(name, *, rank="", path="out/parallel", ext=".pt"):
    """
    Load a tensor from a file.

    Args:
        name (str): The name of the file.
        rank (str, optional): The rank of the current process. Defaults to "".
        path (str, optional): The directory path. Defaults to "out/parallel".
        ext (str, optional): The file extension. Defaults to ".pt".

    Returns:
        torch.Tensor: The loaded tensor.
    """
    return torch.load(get_file(name, rank=rank, path=path, ext=".pt"))


def load_all(name, *, world_size=0, path='out/parallel', ext='.pt'):
    """
    Load tensors from multiple files.

    Args:
        name (str): The name of the file.
        world_size (int, optional): The total number of processes. Defaults to 0.
        path (str, optional): The directory path. Defaults to "out/parallel".
        ext (str, optional): The file extension. Defaults to ".pt".

    Returns:
        Union[torch.Tensor, List[torch.Tensor]]: The loaded tensor or a list of loaded tensors.
    """
    if world_size == -1:
        return load(name, rank='', path=path, ext=ext)
    else:
        return [
            load(name, rank=rank, path=path, ext=ext)
            for rank in range(world_size)
        ]


def save(tensor, name, *, rank="", path="out/parallel", ext=".pt"):
    """
    Save a tensor to a file.

    Args:
        tensor (torch.Tensor): The tensor to be saved.
        name (str): The name of the file.
        rank (str, optional): The rank of the current process. Defaults to "".
        path (str, optional): The directory path. Defaults to "out/parallel".
        ext (str, optional): The file extension. Defaults to ".pt".
    """
    os.makedirs(path, exist_ok=True)
    torch.save(tensor, get_file(name, rank=rank, path=path, ext=".pt"))


def savefig(name, *, path="out/parallel", ext=".pt"):
    """
    Save the current figure to a file.

    Args:
        name (str): The name of the file.
        path (str, optional): The directory path. Defaults to "out/parallel".
        ext (str, optional): The file extension. Defaults to ".pt".
    """
    plt.savefig(get_file(name, rank="", path=path, ext=ext))


def filt(x, sos):
    """
    Apply a biquad filter to the input signal.

    Args:
        x (torch.Tensor): The input signal.
        sos (List[List[float]]): The second-order sections of the filter.

    Returns:
        torch.Tensor: The filtered signal.
    """
    return biquad(biquad(biquad(x, *sos[0]), *sos[1]), *sos[2])


def parse_path(path):
    """
    Parse the input path.

    Args:
        path (str): The input path.

    Returns:
        str: The parsed path.
    """
    if path is None:
        path = "conda"
    if path.startswith("conda"):
        path = path.replace("conda", os.environ["CONDA_PREFIX"])
    elif path.startswith("pwd"):
        path = path.replace("pwd", os.getcwd())
    else:
        path = os.path.join(os.getcwd(), path)
    return path


def auto_path(kw_path="path", make_dir=False):
    """
    Decorator to automatically parse and create directories for file paths.

    Args:
        kw_path (str, optional): The keyword argument name for the file path. Defaults to "path".
        make_dir (bool, optional): Whether to create the directory if it doesn't exist. Defaults to False.

    Returns:
        Callable: The decorator function.
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            if kw_path in kwargs:
                kwargs[kw_path] = parse_path(kwargs[kw_path])
                if make_dir:
                    os.makedirs(kwargs[kwargs[kw_path]], exist_ok=True)
            return func(*args, **kwargs)

        return wrapper

    return decorator


def get_pydict(path, *, filename="metadata", as_class=False):
    """
    Get a Python dictionary from a file.

    Args:
        path (str): The file path.
        filename (str, optional): The name of the file. Defaults to "metadata".
        as_class (bool, optional): Whether to return the dictionary as a DotDict object. Defaults to False.

    Returns:
        Union[dict, DotDict]: The Python dictionary.
    """
    path = parse_path(path)
    filename = filename.replace(".pydict", "") + ".pydict"
    full_path = os.path.join(path, filename)
    d = eval(open(full_path, "r").read())
    if as_class:
        return DotDict(d)
    else:
        return d


def gaussian_perturb(ref, scaled_sigma, scaled_mu, scale=False):
    """
    Generate a Gaussian perturbation based on a reference tensor.

    Args:
        ref (torch.Tensor): The reference tensor.
        scaled_sigma (float): The scaled standard deviation of the Gaussian distribution.
        scaled_mu (float): The scaled mean of the Gaussian distribution.
        scale (bool, optional): Whether to scale the perturbation based on the maximum absolute value of the reference tensor. Defaults to False.

    Returns:
        torch.Tensor: The perturbed tensor.
    """
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
    """
    Convert a verbosity string to an integer value.

    Args:
        verbosity (Union[int, str]): The verbosity level as an integer or string.
        levels (List[Tuple[int, List[str]]]): The list of verbosity levels and their corresponding names.

    Returns:
        int: The converted verbosity level as an integer.

    Raises:
        ValueError: If the verbosity value is not recognized.
    """
    if type(verbosity) == int:
        return verbosity
    elif type(verbosity) == str:
        verbosity = verbosity.lower()
        for level, level_names in levels:
            if verbosity in level_names:
                return level
        raise ValueError(f"Verbose value {verbosity} not recognized")
    else:
        raise ValueError(f"Verbosity must be int or str, got {type(verbosity)}")


def clean_levels(levels):
    """
    Clean and validate the verbosity levels.

    Args:
        levels (List[Union[int, Tuple[int, List[str]]]]): The list of verbosity levels.

    Returns:
        List[Tuple[int, List[str]]]: The cleaned and validated verbosity levels.

    Raises:
        ValueError: If the levels are not in the correct format.
    """
    if levels is None:
        levels = []
        levels.append((0, ["none", "silent"]))
        levels.append((1, ["low", "progress"]))
        levels.append((2, ["medium", "debug"]))
        levels.append((np.inf, ["high", "all"]))
    for i, l in enumerate(levels):
        if type(l) is int:
            levels[i] = (l, [str(l)])
    for i, l in enumerate(levels):
        if type(l) not in [list, tuple] or len(l) != 2:
            raise ValueError("Levels must be list of pairs")
        elif type(l[1]) is not list:
            raise ValueError(f"Level names must be list, got {type(l[1])}")
        elif str(l[0]) not in l[1]:
            l[1].append(str(l[0]))
            if l[0] is np.inf:
                l[1].append("infinity")
    levels = sorted(levels, key=lambda x: x[0])
    return levels


def run_verbosity(*, verbosity, levels):
    """
    Decorator to control the verbosity of a function.

    Args:
        verbosity (Union[int, str]): The verbosity level as an integer or string.
        levels (List[Union[int, Tuple[int, List[str]]]]): The list of verbosity levels.

    Returns:
        Callable: The decorator function.
    """
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


def mem_report(*args, precision=2, sep=", ", rep=None):
    """
    Generate a memory report.

    Args:
        args (float): The memory values to be reported.
        precision (int, optional): The number of decimal places for the memory values. Defaults to 2.
        sep (str, optional): The separator between memory values. Defaults to ", ".
        rep (List[str], optional): The labels for the memory values. Defaults to None.

    Returns:
        str: The memory report.
    """
    filtered_args = []
    if rep is None:
        rep = []
    [rep.append("unknown") for _ in range(len(args) - len(rep))]
    add = lambda x, i: filtered_args.append(x + " (" + rep[i] + ")")
    for i, arg in enumerate(args):
        if 1e18 < arg:
            add(f"{arg/1e18:.{precision}f} EB", i)
        elif 1e15 < arg:
            add(f"{arg/1e15:.{precision}f} PB", i)
        elif 1e12 < arg:
            add(f"{arg/1e12:.{precision}f} TB", i)
        elif 1e9 < arg:
            add(f"{arg/1e9:.{precision}f} GB", i)
        elif 1e6 < arg:
            add(f"{arg/1e6:.{precision}f} MB", i)
        elif 1e3 < arg:
            add(f"{arg/1e3:.{precision}f} KB", i)
        else:
            add(f"{arg:.{precision}f} B", i)
    return sep.join(filtered_args)


def full_mem_report(precision=2, sep=", ", rep=("free", "total"), title=None):
    """
    Generate a full memory report.

    Args:
        precision (int, optional): The number of decimal places for the memory values. Defaults to 2.
        sep (str, optional): The separator between memory values. Defaults to ", ".
        rep (Tuple[str, str], optional): The labels for the memory values. Defaults to ("free", "total").
        title (str, optional): The title of the memory report. Defaults to None.

    Returns:
        str: The full memory report.
    """
    if title is None:
        title = ""
    else:
        title = title + "\n    "
    return title + mem_report(
        *torch.cuda.mem_get_info(), precision=precision, sep=sep, rep=rep
    )


def taper(x, length=100):
    """
    Apply a cosine taper to the ends of a signal.

    Args:
        x (torch.Tensor): The input signal.
        length (int, optional): The length of the taper. Defaults to 100.

    Returns:
        torch.Tensor: The tapered signal.
    """
    return dw.common.cosine_taper_end(x, length)


def downsample_any(u, ratios):
    """
    Downsample a tensor along any dimension.

    Args:
        u (torch.Tensor): The input tensor.
        ratios (List[int]): The downsampling ratios for each dimension.

    Returns:
        torch.Tensor: The downsampled tensor.

    Raises:
        AssertionError: If the number of ratios does not match the number of dimensions.
        AssertionError: If any ratio is not a positive integer.
    """
    assert len(ratios) == len(u.shape), (
        f"downsample_any: len(ratios)={len(ratios)} !="
        f" len(u.shape)={len(u.shape)}"
    )
    assert all(
        [r > 0 and type(r) is int for r in ratios]
    ), f"downsample_any: ratios={ratios} must be positive ints"

    slices = [slice(None, None, r) for r in ratios]
    return u[tuple(slices)]


class SlotMeta(type):
    """
    Metaclass for adding default annotations and __slots__ attribute to a class.

    The metaclass automatically adds default annotations for attributes that are not methods, not in special names, and not already annotated.
    It also creates the __slots__ attribute based on the updated annotations.
    """

    def __new__(cls, name, bases, class_dict):
        # Extract the variable names from the annotations
        try:
            annotated_keys = list(class_dict["__annotations__"].keys())
        except KeyError:
            annotated_keys = []

        # Find attributes that are not methods, not in special names, and not already annotated
        non_annotated_attrs = [
            key
            for key, value in class_dict.items()
            if not (
                callable(value) or key.startswith("__") or key in annotated_keys
            )
        ]

        # Add the default annotations for non-annotated attributes
        for key in non_annotated_attrs:
            class_dict["__annotations__"][key] = Ant[Any, "NOT ANNOTATED"]

            # Optional: Remove the attributes as they'll be defined by __slots__
            class_dict.pop(key, None)

        # Create the __slots__ attribute from updated annotations
        try:
            class_dict["__slots__"] = list(class_dict["__annotations__"].keys())
        except KeyError:
            class_dict["__slots__"] = []

        return super().__new__(cls, name, bases, class_dict)


def idt_print(*args, levels=None, idt="    "):
    """
    Print indented text with different indentation levels.

    Args:
        args (str): The text to be printed.
        levels (List[int], optional): The indentation levels for each text. Defaults to None.
        idt (str, optional): The indentation string. Defaults to "    ".

    Returns:
        str: The indented text.
    """
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
        lines[i] = f"{idt * idt_level}{arg}"
    return "\n".join(lines)


def canonical_tensors(exclude=None, extra=None):
    """
    Get the canonical tensor names.

    Args:
        exclude (List[str], optional): The tensor names to be excluded. Defaults to None.
        extra (List[Union[str, Tuple[str, bool]]], optional): Additional tensor names to be included. Defaults to None.

    Returns:
        List[str]: The canonical tensor names.
    """
    exclude = exclude if exclude else []
    extra = extra if extra else []
    canon = []

    def place(name, init=False):
        if name not in exclude:
            canon.append(name)
            name = name.replace("_record", "")
            if init:
                canon.append(f"{name}_init")
                canon.append(f"{name}_true")

    place("out_record", init=True)
    place("out_filt_record", init=True)
    place("obs_data_filt_record", init=True)
    place("obs_data")
    place("loss")
    place("freqs")
    place("src_amp_y")
    place("rec_loc_y")
    place("src_loc_y")
    for e in extra:
        if type(extra) is tuple:
            place(e[0], init=e[1])
        else:
            place(e)
    return canon


def canonical_reduce(reduce=None, exclude=None, extra=None):
    """
    Get the canonical reduce operations for each tensor.

    Args:
        reduce (Dict[str, str], optional): Custom reduce operations for specific tensors. Defaults to None.
        exclude (List[str], optional): The tensor names to be excluded. Defaults to None.
        extra (Dict[str, str], optional): Additional tensor names and their reduce operations to be included. Defaults to None.

    Returns:
        Dict[str, str]: The canonical reduce operations for each tensor.
    """
    reduce = reduce if reduce else {}
    extra = extra if extra else {}
    exclude = exclude if exclude else []
    canon = canonical_tensors(exclude=exclude, extra=extra)
    default = dict()

    for name in canon:
        has_key = lambda *x: any([e in name for e in x])
        if has_key("cat", "filt", "record"):
            default[name] = "cat"
        elif has_key("stack"):
            default[name] = "stack"
        elif has_key("mean"):
            default[name] = "mean"
        elif has_key("sum"):
            default[name] = "sum"
        else:
            default[name] = None

    return {**default, **reduce}


def see_data(path, cmap='nipy_spectral'):
    """
    Visualize data stored in .pt files.

    Args:
        path (str): The directory path.
        cmap (str, optional): The colormap for visualization. Defaults to 'nipy_spectral'.
    """
    path = os.path.abspath(parse_path(path))
    for file in find_files(path, "*.pt"):
        target = file.replace('.pt', '.jpg')
        if not os.path.exists(target):
            u = torch.load(file).detach().cpu().numpy()
            if len(u.shape) == 2:
                plt.imshow(u, cmap=cmap)
                plt.title(path.split('/')[-1])
                plt.colorbar()
                plt.savefig(file.replace('.pt', '.jpg'))
                plt.clf()


def check_devices(root):
    """
    Check the devices used for storing tensors.

    Args:
        root (str): The root directory path.
    """
    root = parse_path(root)
    pt_path = vco(f'find {root} -type f -name "*.pt"').split('\n')
    headers = ['FILE', 'DEVICE', 'ERROR MESSAGE']
    colors = ['magenta', 'green', 'yellow']
    data = []
    for file in pt_path:
        try:
            u = torch.load(file)
            data.append([file, str(u.device), ''])
            del u
        except Exception as e:
            data.append([file, '', str(e)])
            del u
    s = ctab(data, headers=headers, colors=colors)
    print(s)


def bool_slice(
    *args,
    permute=None,
    none_dims=(),
    ctrl=None,
    strides=None,
    start=None,
    cut=None,
):
    permute = list(permute or range(len(args)))
    permute.reverse()

    # Logic is not correct here for strides, start, cut, etc. TODO: Fix
    strides = strides or [1 for _ in range(len(args))]
    start = start or [0 for _ in range(len(args))]
    cut = cut or [0 for _ in range(len(args))]
    tmp = list(args)
    for i in range(len(strides)):
        if i not in none_dims:
            tmp[i] = (tmp[i] - start[i] - cut[i]) // strides[i]

    args = list(args)
    for i in range(len(args)):
        if i not in none_dims:
            args[i] = args[i] - cut[i]
    # Total number of combinations
    total_combinations = np.prod(
        [e for i, e in enumerate(tmp) if i not in none_dims]
    )

    # Initialize indices
    idx = [
        slice(None) if i in none_dims else start[i] for i in range(len(args))
    ]

    for combo in range(total_combinations):
        print(f'combo={combo}')
        if ctrl is None:
            yield tuple(idx)
        else:
            yield tuple([tuple(idx)]) + (ctrl(idx, args),)

        # Update indices
        for i in permute:
            if i in none_dims:
                continue
            idx[i] += strides[i]
            if idx[i] < args[i]:
                break
            idx[i] = start[i]


def clean_idx(idx, show_colons=True):
    res = [str(e) if e != slice(None) else ':' for e in idx]
    if not show_colons:
        res = [e for e in res if e != ':']
    return f'({", ".join(res)})'


def tensor_summary(t, num=5):
    d = {
        'shape': t.shape,
        'dtype': t.dtype,
        'min': t.min(),
        'max': t.max(),
        'mean': t.mean() if t.dtype == torch.float32 else None,
        'std': t.std() if t.dtype == torch.float32 else None,
        f'top {num} values': torch.topk(t.reshape(-1), num, largest=True)[0],
        f'bottom {num} values': torch.topk(t.reshape(-1), num, largest=False)[
            0
        ],
    }
    s = ''
    for k, v in d.items():
        s += f'{k}\n    {v}\n'
    return s


def pull_data(path):
    d = {}
    keys = [e.replace('.pt', '') for e in os.listdir(path) if e.endswith('.pt')]
    for k in keys:
        d[k] = torch.load(os.path.join(path, k + '.pt'))
    return DotDict(d)
