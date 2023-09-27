import torch
import os
import numpy as np
from obspy.io.segy.segy import _read_segy
import torch
from subprocess import check_output as co
import os
import time
import sys
import torch
from ..swiffer import sco
import re
from warnings import warn
import deepwave as dw
from abc import ABC, abstractmethod
from importlib import import_module
from ..utils import auto_path, parse_path, get_pydict, DotDict
import copy
from warnings import warn
from ..swiffer import iraise, ireraise
from masthay_helpers import prettify_dict
import argparse


def fetch_warn():
    warn(
        'Trying to fetch data from iomt dataset.\n'
        '    This lowers the objectivity of this test.\n'
        '    However, the data has been tested to be the same, but '
        'pride cometh before the fall.\n'
        '    Be careful.\n'
    )


def expand_metadata(meta):
    d = dict()
    for folder, folder_meta in meta.items():
        d[folder] = dict()
        base = {k: v for k, v in folder_meta.items() if type(v) != dict}
        files = {k: v for k, v in folder_meta.items() if type(v) == dict}
        for filename, file_meta in files.items():
            d[folder][filename] = {**base, **file_meta}
            if not d[folder][filename]['url'].endswith('/'):
                d[folder][filename]['url'] += '/'
            if 'filename' not in d[folder][filename].keys():
                d[folder][filename]['filename'] = (
                    filename + '.' + d[folder][filename]['ext']
                )
    return d


def segy_to_torch(
    *,
    input_path,
    output_path,
    device='cpu',
    transpose=False,
    out=sys.stdout,
    print_freq=100,
    **kw,
):
    print('READING in SEGY file "%s"' % input_path, file=out)

    # Read the SEGY file
    stream = _read_segy(input_path)
    # stream = read_segy(input_path, endian='big')

    print('DONE reading SEGY file "%s"' % input_path, file=out)

    num_traces, trace_length = len(stream.traces), len(stream.traces[0].data)
    data_array = np.empty((num_traces, trace_length))

    t = time.time()
    # Loop through each trace in the stream
    for i, trace in enumerate(stream.traces):
        if i == 0:
            print('READING first trace', file=out)
        # Add the trace's data to the list
        data_array[i] = trace.data

        if i % print_freq == 0 and i > 0:
            elapsed = time.time() - t
            avg = elapsed / i
            rem = (num_traces - i) * avg
            print(
                'Elapsed (s): %f, Remaining estimate (s): %f' % (elapsed, rem),
                file=out,
            )

    print('Converting list to pytorch tensor (may take a while)')
    conv = (
        torch.Tensor(data_array).to(device)
        if not transpose
        else torch.Tensor(data_array).transpose(0, 1).to(device)
    )
    torch.save(conv, output_path)


def bin_to_torch(
    *,
    input_path,
    output_path,
    device='cpu',
    transpose=False,
    out=sys.stdout,
    ny,
    nx,
    **kw,
):
    u = torch.from_file(input_path, size=ny * nx)
    torch.save(u.reshape(ny, nx).to(device), output_path)


def any_to_torch(
    *,
    input_path,
    output_path,
    device='cpu',
    transpose=False,
    out=sys.stdout,
    **kw,
):
    if input_path.endswith('.segy') or input_path.endswith('.sgy'):
        segy_to_torch(
            input_path=input_path,
            output_path=output_path,
            device=device,
            transpose=transpose,
            out=out,
            **kw,
        )
    elif input_path.endswith('.bin'):
        bin_to_torch(
            input_path=input_path,
            output_path=output_path,
            device=device,
            transpose=transpose,
            out=out,
            **kw,
        )
    else:
        raise ValueError(f'Unknown file type: {input_path}')


def fetch_data(d, *, path, unzip=True):
    convert_search = dict()
    calls = []
    for folder, info in d.items():
        convert_search[folder] = []

        # make folder if it doesn't exist
        curr_path = os.path.join(path, folder)
        os.makedirs(curr_path, exist_ok=True)

        for file, meta in info.items():
            url = os.path.join(meta['url'], meta['filename'])
            # file_path = f'{folder}/{meta["filename"]}'
            file_path = os.path.join(curr_path, meta['filename'])
            print(f'ATTEMPT: {url} -> {file_path}')
            os.system(f'curl {url} --output {file_path}')
            if unzip and meta['filename'].endswith('.gz'):
                os.system(f'gunzip {file_path}')
                d[folder][file]['filename'] = d[folder][file][
                    'filename'
                ].replace('.gz', '')
            for k, v in meta.items():
                if type(v) == tuple:
                    func = v[0]
                    args = v[1]
                    kwargs = v[2]
                    clos = lambda: func(*args, path=path, **kwargs)
                    calls.append(clos)
    return calls


def convert_data(d, *, path, calls=None):
    for folder, files in d.items():
        for field, meta in files.items():
            curr = os.path.join(path, folder)
            any_to_torch(
                input_path=(os.path.join(curr, meta['filename'])),
                output_path=os.path.join(curr, f'{field}.pt'),
                **meta,
            )
        os.system(
            'rm %s/*.%s'
            % (curr, f' {curr}/*.'.join(['bin', 'sgy', 'segy', 'gz']))
        )
    for call in calls:
        call()


def check_data_installation(path):
    pytorch_files = sco(f'find {path} -name "*.pt"')
    res = {'success': [], 'failure': []}
    if pytorch_files is None or len(pytorch_files) == 0:
        print('NO PYTORCH FILES FOUND')
        return None

    for file in pytorch_files:
        try:
            u = torch.load(file)
            print(f'SUCCESS "{file}" shape={u.shape}')
            res['success'].append(file)
        except:
            print(f'FAILURE "{file}"')
            res['failure'].append(file)
    return res


def store_metadata(*, path, metadata):
    def lean(d):
        omit = ['url', 'filename', 'ext']
        u = {k: v for k, v in d.items() if k not in omit}
        u['source'] = os.path.join(d['url'], d['filename'])
        return u

    res = {}
    for k, v in metadata.items():
        res[k] = {}
        for k1, v1 in v.items():
            res[k][k1] = lean(v1)
        json_path = os.path.join(path, k, 'metadata.json')
        res_str = prettify_dict(res, jsonify=True)
        sep = 80 * '*' + '\n'
        s = sep
        s += f'Storing metadata for {k} in {json_path}\n'
        s += res_str + f'\n{sep}\n'
        print(s)
        with open(json_path, 'w') as f:
            f.write(res_str)


# Extraneous comment
def towed_src(
    *, n_shots, src_per_shot, fst_src, d_src, src_depth, d_intra_shot
):
    res = torch.zeros(n_shots, src_per_shot, 2, dtype=torch.long)
    res[:, :, 1] = src_depth
    for i in range(n_shots):
        for j in range(src_per_shot):
            res[i, j, 0] = fst_src + i * d_src + j * d_intra_shot
    return res


def fixed_rec(*, n_shots, rec_per_shot, fst_rec, d_rec, rec_depth):
    res = torch.zeros(n_shots, rec_per_shot, 2)
    res[:, :, 1] = rec_depth
    res[:, :, 0] = (torch.arange(rec_per_shot) * d_rec + fst_rec).repeat(
        n_shots, 1
    )
    return res


def fetch_and_convert_data(*, subset='all', path=os.getcwd(), check=False):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    datasets = {
        'marmousi': {
            'url': 'https://www.geoazur.fr/WIND/pub/nfs/FWI-DATA/'
            + 'GEOMODELS/Marmousi',
            'ext': 'bin',
            'ny': 2301,
            'nx': 751,
            'dy': 4.0,
            'dx': 4.0,
            'dt': 0.004,
            'd_src': 20,
            'fst_src': 10,
            'src_depth': 2,
            'd_rec': 6,
            'fst_rec': 0,
            'rec_depth': 2,
            'd_intra_shot': 0,
            'freq': 25,
            'peak_time': 1.5 / 25,
            'vp': {},
            'rho': {},
            'obs_data': (create_obs_marm_dw, (), {'device': device}),
        },
        'marmousi2': {
            'url': 'http://www.agl.uh.edu/downloads/',
            'ext': 'segy',
            'vp': {'filename': 'vp_marmousi-ii.segy.gz'},
            'vs': {'filename': 'vs_marmousi-ii.segy.gz'},
            'rho': {'filename': 'density_marmousi-ii.segy.gz'},
        },
        'DAS': {
            'url': 'https://ddfe.curtin.edu.au/7h0e-d392/',
            'ext': 'sgy',
            'das_curtin': {'filename': '2020_GeoLab_WVSP_DAS_wgm.sgy'},
            'geophone_curtin': {
                'filename': '2020_GeoLab_WVSP_geophone_wgm.sgy'
            },
        },
    }
    datasets = expand_metadata(datasets)

    if type(subset) == str:
        subset = [e.strip() for e in subset.split(' ')]

    if path == '' or '/' != path[0]:
        path = os.path.join(os.getcwd(), path)

    if 'all' not in subset and set(subset) != set(datasets.keys()):
        datasets = {k: v for k, v in datasets.items() if k in subset}

    calls = fetch_data(datasets, path=path)
    convert_data(datasets, path=path, calls=calls)
    store_metadata(metadata=datasets, path=path)

    if check:
        res = check_data_installation(path)
        if res is None:
            print('NO PYTORCH FILES FOUND')
        else:
            total = len(res['success']) + len(res['failure'])
            success_head = 'SUCCESS: %d / %d' % (len(res['success']), total)
            print(f'\n{success_head}\n' + '*' * len(success_head))
            print('\n'.join(res['success']))

            failure_head = 'FAILURE: %d / %d' % (len(res['failure']), total)
            print(f'\n{failure_head}\n' + '*' * len(failure_head))
            print('\n'.join(res['failure']))

    return datasets


def get_data(*, field, folder, path=None, check=False):
    if path in [None, 'conda']:
        path = os.path.join(sco('echo $CONDA_PREFIX')[0], 'data')
    elif path == 'pwd':
        path = os.getcwd()
    elif path == '' or path[0] != '/':
        path = os.path.join(os.getcwd(), path)

    full_path = os.path.join(path, folder)
    if os.path.exists(full_path):
        try:
            return torch.load(os.path.join(full_path, f'{field}.pt'))
        except FileNotFoundError:
            print(
                f'File {field}.pt not found in {full_path}'
                + f'\n    Delete {folder} in {path} and try again'
            )
            raise
    # add another
    fetch_and_convert_data(subset=folder, path=path, check=check)
    return torch.load(os.path.join(full_path, f'{field}.pt'))


def get_data2(*, field, path=None, allow_none=False):
    if path is None or path.startswith('conda'):
        if path == 'conda':
            path = 'conda/data'
        else:
            path = path.replace('conda', os.environ['CONDA_PREFIX'])
    elif path.startswith('pwd'):
        path = path.replace('pwd', os.getcwd())
    else:
        path = os.path.join(os.getcwd(), path)

    field_file = os.path.join(path, f'{field}.pt')
    if os.path.exists(path):
        try:
            return torch.load(field_file)
        except FileNotFoundError:
            if allow_none:
                print(f'File {field}.pt not found in {path}, return None')
                return None
            print(
                f'File {field}.pt not found in {path}'
                + f'\n    Delete {path} and try again'
            )
            raise
    subset = path.split('/')[-1]
    dummy_path = '/'.join(path.split('/')[:-1])

    if os.path.exists(path):
        try:
            return torch.load(field_file)
        except FileNotFoundError:
            print(
                f'File {field}.pt not found in {path}'
                + f'\n    Delete {path} and try again'
            )
            raise
    fetch_and_convert_data(subset=subset, path=dummy_path)
    return torch.load(field_file)


def data_path(path):
    def helper(field):
        return {'field': field, 'path': path}

    return helper


def get_data3(*, field, path):
    path = parse_path(path)
    return torch.load(os.path.join(path, f'{field}.pt'))


def field_getter(path):
    def helper(field):
        return get_data3(field=field, path=path)

    return helper


def field_saver(path, verbose=True):
    if verbose:

        def helper(field):
            print(f'Saving {field} to {path}', end='')
            torch.save(field, os.path.join(path, f'{field}.pt'))
            print('SUCCESS')

        return helper
    else:

        def helper(field):
            torch.save(field, os.path.join(path, f'{field}.pt'))

        return helper


def get_metadata(*, path):
    path = parse_path(path)
    return eval(open(f'{path}/metadata.json', 'r').read())


def get_primitives(d):
    prim_list = [int, float, str, bool]
    omit_keys = ['source', 'url', 'filename', 'ext']

    def helper(data, runner):
        for k, v in data.items():
            if k in omit_keys:
                continue
            if type(v) == dict:
                runner = helper(v, runner)
            elif type(v) in prim_list:
                if k in runner.keys() and runner[k] != v:
                    raise ValueError(f'Primitive type mismatch for {k}')
                else:
                    runner[k] = v
        return runner

    return helper(d, {})


def downsample_tensor(tensor, axis, ratio):
    """
    Downsample a torch.Tensor along a given axis by a specific ratio.

    Parameters:
        tensor (torch.Tensor): The input tensor to downsample.
        axis (int): The axis along which to downsample. Must be in range [0, tensor.dim()).
        ratio (int): The downsampling ratio. Must be greater than 0.

    Returns:
        torch.Tensor: The downsampled tensor.
    """

    if ratio <= 0:
        raise ValueError("Ratio must be greater than 0")

    if axis < 0 or axis >= tensor.dim():
        raise ValueError(f"Axis must be in range [0, {tensor.dim()}).")

    slices = [slice(None)] * tensor.dim()
    slices[axis] = slice(None, None, ratio)

    return tensor[tuple(slices)]


def fetch_meta(*, obj):
    parent_module = '.'.join(obj.__module__.split('.')[:-1])
    metadata_module = import_module('.metadata', package=parent_module)
    metadata_func = getattr(metadata_module, 'metadata')
    return metadata_func()


class DataFactory(ABC):
    @auto_path(make_dir=False)
    def __init__(self, *, device=None, src_path, root_out_path, root_path):
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        self.src_path = src_path
        self.parent_path = os.path.dirname(self.src_path)
        self.root_out_path = root_out_path
        self.root_path = root_path
        self.append_path = os.path.relpath(self.path, self.root_path)
        self.out_path = os.path.join(self.root_out_path, self.append_path)
        self.tensors = DotDict(dict())

        py_exists = os.path.exists(f'{self.src_path}/metadata.py')
        pydict_exists = os.path.exists(f'{self.src_path}/metadata.pydict')
        if py_exists:
            cmd = f'python {self.src_path}/metadata.py'
            try:
                os.system(cmd)
            except Exception as e:
                ireraise(
                    e, f'DataFactory Constructor: Error in execution of {cmd}'
                )
        elif not py_exists and not pydict_exists:
            iraise(
                FileNotFoundError,
                f'\n\nNo metadata found in {self.src_path}\n.',
                f'For directories "X" without metadata.py, we populate ',
                'X/metadata.pydict with the metadata from the parent ',
                'prior to generating a DataFactory object',
            )
        self.metadata = get_pydict(self.src_path)

    @abstractmethod
    def _manufacture_data(self, **kw):
        pass

    def manufacture_data(self, **kw):
        self._manufacture_data(**kw)
        self.save_all_tensors()
        self.broadcast_meta()
        self.clear_all_tensors()
        if 'cuda' in self.device:
            torch.cuda.empty_cache()

    def process_web_data(self, **kw):
        d = copy.deepcopy(self.metadata)

        if os.path.exists(self.path):
            print(
                f'{self.path} already exists...ignoring.'
                'If you want to regenerate data, delete this folder '
                'or specify a different path.'
            )
            return
        os.makedirs(self.path, exist_ok=False)

        fields = {
            k: v for k, v in d.items() if type(v) == dict and k != 'derived'
        }
        for k, v in fields.items():
            if 'filename' not in v:
                v['filename'] = k

        def field_url(x):
            url_path = os.path.join(d['url'], fields[x]['filename'])
            if not url_path.endswith('.gz'):
                return url_path + '.' + d['ext']
            else:
                return url_path

        for k, v in fields.items():
            web_data_file = os.path.join(self.path, k) + '.' + d['ext']
            url = field_url(k)
            if url.endswith('.gz'):
                web_data_file += '.gz'
            final_data_file = os.path.join(self.path, k) + '.pt'
            cmd = f'curl {field_url(k)} --output {web_data_file}'
            header = f'ATTEMPT DOWNLOAD: {cmd}'
            stars = len(header) * '*'
            print(f'\n{stars}\nATTEMPT: {cmd}')
            os.system(cmd)
            print(f'SUCCESSFUL DOWNLOAD\n{stars}\n')
            if web_data_file.endswith('.gz'):
                print('About to unzip')
                os.system(f'gunzip {web_data_file}')
                web_data_file = web_data_file.replace('.gz', '')

            any_to_torch(
                input_path=web_data_file,
                output_path=final_data_file,
                **{**d, **v},
            )
            os.system(f'rm {web_data_file}')
            d[k] = torch.load(final_data_file)

        return d

    def save_tensor(self, key):
        path = os.path.join(self.out_path, f'{key}.pt')
        torch.save(getattr(self.tensors, key), path)

    def save_all_tensors(self):
        for k in self.tensors.__dict__.keys():
            self.save_tensor(k)

    def clear_tensor(self, key):
        delattr(self.tensors, key)

    def clear_all_tensors(self):
        delattr(self, 'tensors')

    def broadcast_meta(self):
        submeta = DataFactory.get_derived_meta(meta=self.metadata)
        if submeta is None:
            return None
        for k, v in submeta.items():
            os.makedirs(f'{self.src_path}/{k}', exist_ok=True)
            with open(f'{self.src_path}', 'w') as f:
                f.write(prettify_dict(v))

    @staticmethod
    def get_derived_meta(*, meta):
        if 'derived' not in meta:
            return None
        base_items = {k: v for k, v in meta.items() if type(v) != dict}
        derived = meta['derived']
        common = {**base_items, **derived.get('common', {})}
        if 'common' in derived:
            del derived['common']
        for k, v in derived.items():
            derived[k] = {**common, **v}
        return derived

    @staticmethod
    def manufacture_all(*, root, root_out_path):
        for dir_path, dir_names, file_names in os.walk(root):
            if dir_path != root:
                DataFactoryTree.deploy_factory(
                    root=root, root_out=root_out_path, src_path=dir_path
                )

    @staticmethod
    def deploy_factory(*, root, root_out_path, src_path):
        if not os.path.exists(
            f'{src_path}/metadata.pydict'
        ) and not os.path.exists(f'{src_path}/metadata.py'):
            iraise(FileNotFoundError, f'No metadata found in {src_path}')
        if not os.path.exists(f'{src_path}/factory.py'):
            iraise(FileNotFoundError, f'No factory.py found in {src_path}')

        cmd = (
            f'python {src_path}/factory.py --root {root} --root_out'
            f' {root_out_path}'
        )
        try:
            os.system(cmd)
        except Exception as e:
            msg = str(e)
            iraise(type(e), f'Error in execution of {cmd}', msg)

    @classmethod
    def cli_construct(cls, *, device=None, src_path, **kw):
        parser = argparse.ArgumentParser()
        parser.add_argument('--root', type=str, required=True)
        parser.add_argument('--root_out', type=str, required=True)
        args = parser.parse_args()
        return cls(
            device=device,
            src_path=src_path,
            root_out_path=args.root_out,
            root_path=args.root,
            **kw,
        )


# class DataFactoryMeta(DataFactory):
#     def __init__(self, *, path, device=None):
#         super().__init__(path=path)
#         if device is None:
#             self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
#         else:
#             self.device = device
#         # parent_module = '.'.join(self.__module__.split('.')[:-1])
#         # metadata_module = import_module('.metadata', package=parent_module)
#         # metadata_func = getattr(metadata_module, 'metadata')
#         # self.metadata = metadata_func()
#         self.metadata = fetch_meta(obj=self)

#     # def manufacture_data(self):
#     #     d = self._manufacture_data(metadata=self.metadata)
#     #     with open(os.path.join(self.path, 'metadata.pydict'), 'w') as f:
#     #         f.write(prettify_dict(self.metadata))
#     #     return d

#     @staticmethod
#     def get_derived_meta(*, meta):
#         if 'derived' not in meta:
#             return None
#         base_items = {k: v for k, v in meta.items() if type(v) != dict}
#         derived = meta['derived']
#         common = {**base_items, **derived.get('common', {})}
#         if 'common' in derived:
#             del derived['common']
#         for k, v in derived.items():
#             derived[k] = {**common, **v}
#         return derived


class DataFactoryTree(DataFactory):
    """
    data: Stores all data, with tensors being evaluated now + all the metadata
    """

    # src_dir = os.path.dirname(src_path)

    # pydict_exists = os.path.exists(os.path.join(src_dir, 'metadata.pydict'))
    # py_exists = os.path.exists(os.path.join(src_dir, 'metadata.py'))
    # if not py_exists:
    #     if not pydict_exists:
    #         raise FileNotFoundError(
    #             'FATAL: Either metadata.pydict or metadata.py must exist'
    #             f' in\n    {src_dir}\n'
    #         )
    #     else:
    #         metadata = get_pydict(src_dir)
    # else:
    #     metadata = fetch_meta(obj=self)
    # if not os.path.exists(os.path.join(src_dir, 'factory.py')):
    #     iraise(
    #         ValueError,
    #         f'FATAL: factory.py must exist in directory {src_dir}',
    #     )
    # else:
    #     factory = import_module('.factory', package=self.__module__)
    #     factory_main = getattr(factory, 'main')
    #     factory_main()

    def get_parent_meta(self):
        parent_abs_path = '/'.join(self.fpath.split('/')[:-1])
        pydict_exists = os.path.exists(
            os.path.join(parent_abs_path, 'metadata.pydict')
        )
        py_exists = os.path.exists(os.path.join(parent_abs_path, 'metadata.py'))

        if pydict_exists and not py_exists:
            try:
                return get_pydict(parent_abs_path)
            except Exception as e:
                ireraise(
                    e,
                    f'Error in {parent_abs_path}/metadata.pydict\n',
                    f'IOMT USER RESPONSIBILTY: "python {parent_abs_path}',
                    f'/metadata.py" should create a file at {parent_abs_path}',
                    '/metadata.pydict that is a valid python dictionary.',
                )
        elif py_exists:
            os.system(f'python {parent_abs_path}/metadata.py')
            return get_pydict(parent_abs_path)
        else:
            iraise(
                FileNotFoundError,
                f'No metadata found in {parent_abs_path}',
            )

    class LocalFactory(DataFactory):
        def __init__(
            self, *, path, device=None, src_path, root_out_path, root_path
        ):
            super().__init__(path=path, device=device)
            self.src_path = src_path
            self.root_out_path = root_out_path
            self.root_path = root_path
            self.append_path = os.path.relpath(self.path, self.root_path)
            self.out_path = os.path.join(self.root_out_path, self.append_path)
