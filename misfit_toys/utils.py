import matplotlib.pyplot as plt
import numpy as np
import torch

from typing import Annotated as Ant, Any

import deepwave as dw
import os

from masthay_helpers.global_helpers import find_files, vco, ctab, DotDict
import torch.distributed as dist
from torchaudio.functional import biquad


def setup(rank, world_size, port=12355):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def get_file(name, *, rank="", path="out/parallel", ext=".pt"):
    ext = "." + ext.replace(".", "")
    name = name.replace(ext, "")
    if rank != "":
        rank = f"_{rank}"
    return os.path.join(os.path.dirname(__file__), path, f"{name}{rank}{ext}")


def load(name, *, rank="", path="out/parallel", ext=".pt"):
    return torch.load(get_file(name, rank=rank, path=path, ext=".pt"))


def load_all(name, *, world_size=0, path='out/parallel', ext='.pt'):
    if world_size == -1:
        return load(name, rank='', path=path, ext=ext)
    else:
        return [
            load(name, rank=rank, path=path, ext=ext)
            for rank in range(world_size)
        ]


def save(tensor, name, *, rank="", path="out/parallel", ext=".pt"):
    os.makedirs(path, exist_ok=True)
    torch.save(tensor, get_file(name, rank=rank, path=path, ext=".pt"))


def savefig(name, *, path="out/parallel", ext=".pt"):
    plt.savefig(get_file(name, rank="", path=path, ext=ext))


# def taper(x):
#     # Taper the ends of traces
#     return dw.common.cosine_taper_end(x, 100)


def filt(x, sos):
    return biquad(biquad(biquad(x, *sos[0]), *sos[1]), *sos[2])


def parse_path(path):
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
    path = parse_path(path)
    filename = filename.replace(".pydict", "") + ".pydict"
    full_path = os.path.join(path, filename)
    d = eval(open(full_path, "r").read())
    if as_class:
        return DotDict(d)
    else:
        return d


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
        raise ValueError(f"Verbose value {verbosity} not recognized")
    else:
        raise ValueError(f"Verbosity must be int or str, got {type(verbosity)}")


def clean_levels(levels):
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
    if title is None:
        title = ""
    else:
        title = title + "\n    "
    return title + mem_report(
        *torch.cuda.mem_get_info(), precision=precision, sep=sep, rep=rep
    )


def taper(x, length=100):
    return dw.common.cosine_taper_end(x, length)


def downsample_any(u, ratios):
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
    def __new__(cls, name, bases, class_dict):
        # Extract the variable names from the annotations
        try:
            annotated_keys = list(class_dict["__annotations__"].keys())
        except KeyError:
            annotated_keys = []

        # Find attributes that are not methods, not in special names and not already annotated
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

        # Create the __slots__ attribute from updated annotationsi
        try:
            class_dict["__slots__"] = list(class_dict["__annotations__"].keys())
        except KeyError:
            class_dict["__slots__"] = []

        return super().__new__(cls, name, bases, class_dict)


# class CombinedMeta(SlotMeta, ABCMeta):
#     pass


def idt_print(*args, levels=None, idt="    "):
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
