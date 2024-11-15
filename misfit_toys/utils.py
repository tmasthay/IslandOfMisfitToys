"""
Utility functions, many of which are deprecated or not used in current state of code.
"""

import glob
import os
import socket
from typing import Annotated as Ant
from typing import Any

import deepwave as dw

# Rest of the code...
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from mh.core import DotDict, DotDictImmutable, exec_imports
from mh.core_legacy import ctab, find_files, vco
from omegaconf import DictConfig, OmegaConf
from returns.curry import curry
from torchaudio.functional import biquad

from misfit_toys.types import PickleUnaryFunction as PUF


def find_available_port(start_port, max_attempts=5):
    """
    Finds an available port starting from the given start_port.

    Args:
        start_port (int): The port number to start searching from.
        max_attempts (int, optional): The maximum number of attempts to find an available port. Defaults to 5.

    Returns:
        int: The first available port number found.

    Raises:
        OSError: If an error occurs while attempting to bind to a port.

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


def setup(rank, world_size, port=12358):
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
    Cleans up the process group.

    This function destroys the process group using the `dist.destroy_process_group()` method.

    Returns:
        None
    """
    print('yo', flush=True)
    dist.destroy_process_group()
    print('destroyed', flush=True)


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
    if name == 'path_record':
        return
    os.makedirs(path, exist_ok=True)
    filename = get_file(name, rank=rank, path=path, ext=ext)
    torch.save(tensor, filename)
    if not os.path.exists(filename):
        raise ValueError(f'{name}, {type(tensor)}, {filename}')


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

    def v2i(x):
        verbosity_str_to_int(verbosity=x, levels=levels)

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

    def add(x, i):
        filtered_args.append(x + " (" + rep[i] + ")")

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
    """Metaclass that adds default annotations and __slots__ attribute to a class.

    This metaclass is used to automatically add default annotations and define the __slots__
    attribute for a class. It extracts the variable names from the annotations and finds
    attributes that are not methods, not in special names, and not already annotated. It then
    adds default annotations for these non-annotated attributes and removes them from the class
    dictionary. Finally, it creates the __slots__ attribute from the updated annotations.

    Attributes:
        None

    Methods:
        __new__(cls, name, bases, class_dict): Creates a new class with default annotations and __slots__ attribute.

    Example usage:
        class MyClass(metaclass=SlotMeta):
            attr1: int
            attr2: str
            ...
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
            # class_dict["__annotations__"] = Ant[Any, "NOT ANNOTATED"]

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

        def has_key(x):
            # original was lambda *x: ... --- unsure if this works,
            #     THIS CODE IS DEPRECATED ANYWAY
            any([e in name for e in x])

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
            # del u
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
    verbose=False,
):
    """
    Generate boolean slices based on the given arguments.

    Args:
        *args: Variable length arguments representing the dimensions of the boolean slices.
        permute (list): A list specifying the order in which the dimensions should be permuted.
        none_dims (tuple): A tuple containing the indices of dimensions that should be treated as None.
        ctrl (function): A function that takes in the indices and arguments and returns a boolean value.
        strides (list): A list specifying the strides for each dimension.
        start (list): A list specifying the starting indices for each dimension.
        cut (list): A list specifying the cut values for each dimension.
        verbose (bool): A boolean value indicating whether to print verbose output.

    Yields:
        tuple: A tuple containing the boolean slice indices and the result of the control function.

    """
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
    none_dims = [i if i >= 0 else len(args) + i for i in none_dims]
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

    if ctrl is None:

        def ctrl_default(*args):
            return True

        ctrl = ctrl_default

    for combo in range(total_combinations):
        res = tuple([tuple(idx)]) + (ctrl(idx, args),)
        if verbose:
            print(f'bool_slice: {combo} -> {res}')
        yield res

        # Update indices
        for i in permute:
            if i in none_dims:
                continue
            idx[i] += strides[i]
            if idx[i] < args[i]:
                break
            idx[i] = start[i]


def clean_idx(idx, show_colons=True):
    """
    Cleans up the given index by converting it to a string representation.

    Args:
        idx (list): The index to clean up.
        show_colons (bool, optional): Whether to include colons in the string representation.
            Defaults to True.

    Returns:
        str: The cleaned up string representation of the index.
    """
    res = [str(e) if e != slice(None) else ':' for e in idx]
    if not show_colons:
        res = [e for e in res if e != ':']
    return f'({", ".join(res)})'


@curry
def tensor_summary(t, num=5, inc='all', exc=None):
    """
    Summarizes the properties of a tensor.

    Args:
        t (torch.Tensor): The input tensor.
        num (int, optional): The number of top and bottom values to include in the summary. Defaults to 5.
        inc (list or str, optional): The properties to include in the summary. Defaults to 'all'.
        exc (list, optional): The properties to exclude from the summary. Defaults to None.

    Returns:
        str: A string containing the summary of the tensor properties.
    """
    num = min(num, t.numel())
    if inc == 'all':
        inc = [
            'shape',
            'dtype',
            'min',
            'max',
            'mean',
            'std',
            f'top {num} values',
            f'bottom {num} values',
        ]
    inc = set(inc).difference(exc or [])
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
    d = {k: v for k, v in d.items() if k in inc}
    s = ''
    for k, v in d.items():
        s += f'{k}:    {v}\n'
    return s


def pull_data(path):
    """
    Loads data from the specified path.

    Args:
        path (str): The path to the directory containing the data files.

    Returns:
        DotDict: A dictionary-like object containing the loaded data.

    """
    d = {}
    keys = [e.replace('.pt', '') for e in os.listdir(path) if e.endswith('.pt')]
    for k in keys:
        d[k] = torch.load(os.path.join(path, k + '.pt'))
    return DotDict(d)


def mean_filter_1d(y, kernel_size):
    """
    Applies a 1-dimensional mean filter to the input tensor.

    Args:
        y (torch.Tensor): The input tensor to be filtered.
        kernel_size (int): The size of the kernel for the mean filter.

    Returns:
        torch.Tensor: The filtered output tensor.

    Raises:
        None

    Examples:
        >>> input_tensor = torch.tensor([1, 2, 3, 4, 5])
        >>> filtered_tensor = mean_filter_1d(input_tensor, 3)
        >>> print(filtered_tensor)
        tensor([2., 3., 4., 3., 2.])
    """
    num_elems = y.numel() // y.shape[-1]
    input_tensor = y.reshape(num_elems, 1, y.shape[-1])
    kernel = torch.ones((kernel_size,)).unsqueeze(0).unsqueeze(0) / kernel_size
    kernel = kernel.to(input_tensor.device)

    padding_size = kernel_size // 2
    if kernel_size % 2 == 0:
        left_padding, right_padding = padding_size, padding_size - 1
    else:
        left_padding, right_padding = padding_size, padding_size

    if padding_size > 0:
        input_tensor = F.pad(
            input_tensor, (left_padding, right_padding), mode='reflect'
        )

    mean_filter = torch.nn.Conv1d(
        1, 1, kernel_size, bias=False, padding=0, groups=1
    )
    mean_filter.weight.data = kernel
    mean_filter.weight.requires_grad = False

    output_tensor = mean_filter(input_tensor)
    output_tensor = output_tensor.reshape(y.size())

    return output_tensor


def get_tensors(path, device='cpu'):
    """
    Loads tensors from files in the specified path and returns them as a dictionary.

    Args:
        path (str): The path to the directory containing the tensor files.
        device (str, optional): The device to move the loaded tensors to. Defaults to 'cpu'.

    Returns:
        dict: A dictionary where the keys are the filenames (without the file extension) and the values are the loaded tensors.
    """
    d = DotDict({})
    files = [e[:-3] for e in os.listdir(path) if e.endswith('.pt')]
    for f in files:
        d[f] = torch.load(f'{path}/{f}.pt')
        if device is not None:
            d[f] = d[f].to(device)
    return d


def d2cpu(x):
    """
    Moves a tensor from GPU to CPU and detaches it from the computation graph.

    Args:
        x (torch.Tensor): The input tensor.

    Returns:
        torch.Tensor: The tensor moved to CPU and detached from the computation graph.
    """
    return x.detach().cpu()


def chunk_params(rank, world_size, *, params, chunk_keys):
    """
    Chunks the parameters based on the rank and world size.

    Args:
        rank (int): The rank of the current process.
        world_size (int): The total number of processes.
        params (dict): A dictionary containing the parameters.
        chunk_keys (list): A list of keys to chunk.

    Returns:
        dict: A dictionary containing the chunked parameters.

    """
    for k in chunk_keys:
        params[k].p.data = torch.chunk(params[k].p.data, world_size)[rank]
    return params


def chunk_tensors(rank, world_size, *, data, chunk_keys):
    """
    Chunks the tensors based on the rank and world size.

    Args:
        rank (int): The rank of the current process.
        world_size (int): The total number of processes.
        data (dict): A dictionary containing the tensors.
        chunk_keys (list): A list of keys to chunk.

    Returns:
        dict: A dictionary containing the chunked tensors.

    """
    for k in chunk_keys:
        data[k] = torch.chunk(data[k], world_size)[rank].to(rank)
    return data


def deploy_data(rank, data):
    """
    Deploys the data to the specified rank.

    Args:
        rank (int): The rank to deploy the data to.
        data (dict): A dictionary containing the data.

    Returns:
        dict: A dictionary containing the deployed data.

    """
    for k, v in data.items():
        if k != 'meta':
            data[k] = v.to(rank)
    return data


def chunk_and_deploy(rank, world_size, *, data, chunk_keys):
    """
    Chunks and deploys the data based on the rank and world size.

    Args:
        rank (int): The rank of the current process.
        world_size (int): The total number of processes.
        data (dict): A dictionary containing the data.
        chunk_keys (dict): A dictionary containing the keys to chunk.

    Returns:
        dict: A dictionary containing the chunked and deployed data.

    """
    data = chunk_tensors(
        rank, world_size, data=data, chunk_keys=chunk_keys['tensors']
    )
    data = chunk_params(
        rank, world_size, params=data, chunk_keys=chunk_keys['params']
    )
    data = deploy_data(rank, data)
    return data


def read_and_chunk(*, path, rank, world_size, chunk_keys, remap=None, **kw):
    """
    Read data from a given path, perform remapping and transformations, and chunk the data for distributed processing.

    Args:
        path (str): The path to the data.
        rank (int): The rank of the current process.
        world_size (int): The total number of processes.
        chunk_keys (list): A list of keys to chunk the data.
        remap (dict, optional): A dictionary to remap the keys of the data. Defaults to None.
        **kw: Additional keyword arguments for transformations.

    Returns:
        dict: The processed data.

    """
    remap = remap or {}
    d = get_tensors(path)
    for k, v in remap.items():
        d[remap[k]] = d.pop(k)
    for k, v in kw.items():
        d[k] = v(d[k])
    d.meta = DotDict(eval(open(f'{path}/metadata.pydict', 'r').read()))
    chunk_and_deploy(rank, world_size, data=d, chunk_keys=chunk_keys)
    return d


def get_gpu_memory(rank):
    """
    Retrieves the GPU memory information for the specified GPU device.

    Args:
        rank (int): The rank of the GPU device.

    Returns:
        dict: A dictionary containing the GPU memory information.
            - 'rank' (int): The rank of the GPU device.
            - 'total_memory_GB' (float): The total GPU memory in gigabytes (GB).
            - 'allocated_memory_GB' (float): The currently allocated GPU memory in gigabytes (GB).
            - 'cached_memory_GB' (float): The GPU memory reserved for caching in gigabytes (GB).
            - 'available_memory_GB' (float): The available GPU memory in gigabytes (GB).
    """
    torch.cuda.synchronize()  # Synchronizes all kernels and operations to ensure correct memory readings
    total_memory = torch.cuda.get_device_properties(rank).total_memory
    allocated_memory = torch.cuda.memory_allocated(rank)
    cached_memory = torch.cuda.memory_reserved(rank)
    available_memory = total_memory - max(allocated_memory, cached_memory)

    return {
        'rank': rank,
        'total_memory_GB': total_memory / (1024**3),  # Convert to GB
        'allocated_memory_GB': allocated_memory / (1024**3),
        'cached_memory_GB': cached_memory / (1024**3),
        'available_memory_GB': available_memory / (1024**3),
    }


def apply_builder(lcl, gbl):
    """
    Applies the builder to create and return an object.

    Args:
        lcl (dict): The local namespace dictionary.
        gbl (dict): The global namespace dictionary.

    Returns:
        object: The created object.
    """
    builder = lcl.builder
    print(builder, flush=True)
    if 'func' in builder.keys():
        args, kwargs = builder.func(gbl, *builder.args, **builder.kw)
    else:
        args = builder.get('args', [])
        kwargs = builder.get('kw', {}) or builder.get('kwargs', {})
    obj = lcl.type(*args, **kwargs)
    return obj


def apply(
    lcl: dict,
    relax: bool = True,
    call_key: str = '__call__',
    allow_implicit: bool = False,
    protected_keys: list = None,
):
    """
    Applies the given local variables (`lcl`) to a runtime function.

    Args:
        lcl (dict): The local variables to be applied.
        relax (bool, optional): Whether to relax the requirement of having '__call__' as a key in `lcl`.
            If set to True and '__call__' is not found in `lcl`, the function will simply return `lcl`.
            If set to False and '__call__' is not found in `lcl`, a ValueError will be raised.

    Returns:
        The result of applying the local variables to the runtime function.

    Raises:
        ValueError: If '__call__' is not found in `lcl` and `relax` is set to False.
        RuntimeError: If an error occurs during the application process.

    """
    protected_keys = protected_keys or ['__tmp__']
    try:
        if call_key not in lcl.keys() and relax:
            return lcl
        elif call_key not in lcl.keys() and not relax:
            raise ValueError(
                "To apply lcl, we need runtime_func to be a key "
                f"in lcl, but it is not. lcl.keys() = {lcl.keys()}"
            )
        all_keys = set(lcl.keys())
        args = lcl.get('args', [])
        if 'kwargs' in all_keys and 'kw' in all_keys:
            raise ValueError(
                f'Ambigous kwargs, both kwargs and kw are present, {lcl=}'
            )
        kwargs = lcl.get('kwargs', {}) or lcl.get('kw', {})
        implicit_kwargs = all_keys.difference(
            set(['args', 'kwargs', 'kw', call_key, *protected_keys])
        )
        if allow_implicit:
            if len(implicit_kwargs & set(kwargs.keys())) > 0:
                raise ValueError(
                    'Implicit kwargs overlap with explicit kwargs\n   '
                    f' {lcl=}\n    {implicit_kwargs=}\n    {kwargs=}'
                )
            kwargs.update({k: lcl[k] for k in implicit_kwargs})

        for i, e in enumerate(args):
            if isinstance(e, DotDict) or isinstance(e, dict):
                args[i] = apply(
                    e,
                    relax=True,
                    call_key=call_key,
                    allow_implicit=allow_implicit,
                )

        for k, v in kwargs.items():
            if isinstance(v, DotDict) or isinstance(v, dict):
                kwargs[k] = apply(
                    v,
                    relax=True,
                    call_key=call_key,
                    allow_implicit=allow_implicit,
                )

        # keys = set(kwargs.keys())
        # is_reducible = keys.issubset(
        #     set(['args', 'kwargs', 'kw', call_key])
        # )
        if call_key in kwargs.keys():
            kwargs = apply(kwargs, relax=True, call_key=call_key)
        lcl_callback = getattr(lcl, call_key)
        lcl = lcl_callback(*args, **kwargs)
        return lcl
    except Exception as e:
        v = RuntimeError(
            f'Error in apply\n{args=}\n{kwargs=}\n{lcl=}\nPrevious error "{e}"'
            ' full was the cause of this exception.\nScroll up past "The above'
            ' exception was the direct cause of the following exception" to'
            ' see the full error in the stack trace.'
        )
        raise v from e


def apply_all(
    lcl,
    *, 
    relax=True,
    exc=None,
    call_key='__call__',
    allow_implicit=False,
    protected_keys=None,
):
    """
    Recursively applies the `apply` function to all values in a dictionary or DotDict.

    Args:
        lcl (dict or DotDict): The dictionary or DotDict to apply the `apply` function to.
        relax (bool, optional): Whether to relax the application of `apply` function. Defaults to True.
        exc (list, optional): List of keys to exclude from applying the `apply` function. Defaults to None.

    Returns:
        dict or DotDict: The modified dictionary or DotDict after applying the `apply` function.
    """
    if isinstance(lcl, DotDict) or isinstance(lcl, dict):
        if call_key in lcl.keys():
            return apply(
                lcl,
                relax=relax,
                call_key=call_key,
                allow_implicit=allow_implicit,
                protected_keys=protected_keys,
            )

    valid_items = [(k, v) for k, v in lcl.items() if k not in (exc or [])]
    for k, v in valid_items:
        if isinstance(v, DotDict) or isinstance(v, dict):
            lcl[k] = apply_all(
                v,
                relax=relax,
                exc=exc,
                call_key=call_key,
                allow_implicit=allow_implicit,
                protected_keys=protected_keys,
            )
    return lcl


def runtime_reduce(
    config: DotDict,
    *,
    relax: bool = False,
    call_key: str = '__call__',
    self_key: str = 'self',
    allow_implicit: bool = False,
    protected_keys: list = None,
    exc: list = None,
):
    config = config.self_ref_resolve(self_key=self_key, relax=relax)
    config = apply_all(
        config,
        relax=relax,
        call_key=call_key,
        exc=exc,
        allow_implicit=allow_implicit,
        protected_keys=protected_keys,
    )
    return config


def resolve(c: DotDict, relax) -> DotDict:
    """
    Resolves the given DotDict object by executing imports and resolving self-references.

    Args:
        c (DotDict): The DotDict object to be resolved.
        relax: A flag indicating whether to relax the resolution process.

    Returns:
        DotDict: The resolved DotDict object.
    """
    c = exec_imports(c)
    c.self_ref_resolve(gbl=globals(), lcl=locals(), relax=relax)
    return c


def git_dump_info(exc=None):
    """
    Retrieves information about the Git repository.

    Args:
        exc (list, optional): List of directories or files to exclude from the untracked files list. Defaults to None.

    Returns:
        str: A string containing information about the Git repository.
    """
    exc = exc or ['outputs', 'multirun', '__pycache__']
    s = ''
    s += f'HASH: {vco("git rev-parse HEAD")}\n'
    s += f'BRANCH: {vco("git rev-parse --abbrev-ref HEAD")}\n\n'
    untracked_files_cmd = 'git ls-files --others'
    for e in exc:
        untracked_files_cmd += f' | grep -v "^{e}"'
    s += f'UNTRACKED FILES: {vco(untracked_files_cmd)}\n\n'
    s += 80 * '*' + '\n'
    s += f'DIFF: {vco("git diff")}\n'
    s += 80 * '*' + '\n'

    return s


# def vco(cmd):
#     """
#     Executes a shell command and returns the output.

#     Args:
#         cmd (str): The shell command to execute.

#     Returns:
#         str: The output of the shell command.
#     """
#     # Implementation of the vco function goes here
#     pass


# Other functions in the highlighted section with their docstrings go here


def all_detached_cpu(d: DotDict):
    for k, v in d.items():
        if isinstance(v, DotDict):
            all_detached_cpu(v)
        elif isinstance(v, torch.Tensor):
            d[k] = v.detach().cpu()

    return d


# the following two functions were created in the tri_gpu branch, which was merged into feature/source
# a bit prematurely. These functions can look a little confusing at first, but they are super useful.
# Point is to encode how to preprocess the config by passing a subdictionary and then "eating" that subdictionary
# up after the initialization (since that is the only scope it is needed in).
def self_read_cfg(cfg: DictConfig, *, read_key='read', **kw):
    if 'read_key' in cfg:
        read_key = cfg.read_key
    relax = cfg[read_key].get('relax', False)
    if read_key not in cfg.keys():
        raise ValueError(f"Key {read_key} not found in cfg")
    if not isinstance(cfg, DotDict):
        c = DotDict(OmegaConf.to_container(cfg[read_key], resolve=True))
    else:
        c = cfg[read_key]
    self_read: PUF = apply_all(exec_imports(c).self_ref_resolve(relax=relax))
    return self_read(cfg, **kw)


def preprocess_cfg(
    x: DictConfig,
    *,
    no_self_ref=None,
    no_apply=None,
    remove_self_read_key=None,
    mutable=True,
    compose=None,
    eat_key=None,
) -> DotDict:
    container = DotDict if mutable else DotDictImmutable

    d = exec_imports(container(OmegaConf.to_container(x, resolve=True)))
    no_self_ref = no_self_ref or []
    no_apply = no_apply or []
    for k, v in d.items():
        if k not in no_self_ref and isinstance(v, DotDict):
            d[k] = v.self_ref_resolve()
        if k not in no_apply:
            d[k] = apply_all(d[k], relax=True)

    if remove_self_read_key:
        del d[remove_self_read_key]
    if eat_key is not None:
        d = d[eat_key]
    return d if compose is None else compose(d)


def select_best_gpu():
    if not torch.cuda.is_available():
        return "cpu"

    best_gpu = "cuda:0"
    max_free_vram = 0

    for i in range(torch.cuda.device_count()):
        free_vram = torch.cuda.memory_reserved(i) - torch.cuda.memory_allocated(
            i
        )
        if free_vram > max_free_vram:
            max_free_vram = free_vram
            best_gpu = f"cuda:{i}"

    return best_gpu

def tslice(u, *, dims=None, idxs=None):
    if dims is None:
        return u
    
    dims = [e if e > 0 else len(u.shape) + e for e in dims]
    idxs = idxs or [-1 for _ in range(len(dims))]
    assert len(idxs) == len(dims)
    final_idx = [slice(None) if i not in dims else idxs[dims.index(i)] for i in range(len(u.shape))]
    return u[tuple(final_idx)]
    