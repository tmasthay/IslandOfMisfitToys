from ...utils import auto_path, get_pydict, SlotMeta, DotDict, iraise
from ...data.dataset import *
from .models import Param, ParamConstrained

from deepwave.common import vpvsrho_to_lambmubuoyancy as get_lame

from torchaudio.functional import biquad
from typing import Annotated as Ant, Optional as Opt, Union, Callable as Call
from scipy.ndimage import gaussian_filter
from dataclasses import dataclass, field
import json

from masthay_helpers import printj


class SeismicProp(torch.nn.Module, metaclass=SlotMeta):
    obs_data: Ant[torch.Tensor, 'Observed data']
    src_amp_y: Ant[torch.Tensor, 'Source amplitude, y component']
    src_amp_x: Opt[Ant[torch.Tensor, 'Source amplitude, x component']]
    src_loc_y: Ant[torch.Tensor, 'Source locations']
    rec_loc_y: Ant[torch.Tensor, 'Receiver locations']
    vp: Ant[torch.Tensor, 'Initial P velocity model']
    vs: Opt[Ant[torch.Tensor, 'Initial S velocity model']]
    rho: Opt[Ant[torch.Tensor, 'Initial density model']]
    vp_true: Opt[Ant[torch.Tensor, 'True P velocity model']]
    vs_true: Opt[Ant[torch.Tensor, 'True S velocity model']]
    rho_true: Opt[Ant[torch.Tensor, 'True density model']]
    src_amp_y_true: Opt[Ant[torch.Tensor, 'True source amplitude, y component']]
    src_amp_x_true: Opt[Ant[torch.Tensor, 'True source amplitude, x component']]
    nx: Ant[int, 'Number of x grid points']
    ny: Ant[int, 'Number of y grid points']
    nt: Ant[int, 'Number of time steps']
    dx: Ant[float, 'Grid spacing in x']
    dy: Ant[float, 'Grid spacing in y']
    dt: Ant[float, 'Time step']
    n_shots: Ant[int, 'Number of shots']
    src_per_shot: Ant[int, 'Number of sources per shot']
    rec_per_shot: Ant[int, 'Number of receivers per shot']
    freq: Ant[float, 'Source frequency']
    extra_forward_args: Ant[dict, 'Extra arguments to forward pass']
    metadata: Ant[dict, 'Metadata']
    custom: Ant[dict, 'Custom data']

    @auto_path(make_dir=False)
    def __init__(
        self,
        *,
        path: Ant[str, 'Path to data'],
        extra_forward_args: Opt[Ant[dict, 'Extra forward args']] = None,
        obs_data: Ant[Union[str, torch.Tensor], 'obs_data'] = None,
        src_amp_y: Ant[Union[str, torch.Tensor], 'Source amp. y'] = None,
        src_loc_y: Ant[Union[str, torch.Tensor], 'Source locations'] = None,
        rec_loc_y: Ant[Union[str, torch.Tensor], 'Receiver locations'] = None,
        vp_init: Ant[
            Union[str, torch.Tensor], 'Initial P velocity model'
        ] = None,
        src_amp_x: Opt[Ant[Union[str, torch.Tensor], 'Source amp. x']] = None,
        vs_init: Opt[Ant[Union[str, torch.Tensor], 'Init S vel']] = None,
        rho_init: Opt[Ant[Union[str, torch.Tensor], 'Init density']] = None,
        vp_true: Opt[Ant[Union[str, torch.Tensor], 'True P vel']] = None,
        vs_true: Opt[Ant[Union[str, torch.Tensor], 'True S vel']] = None,
        rho_true: Opt[Ant[Union[str, torch.Tensor], 'True density ']] = None,
        src_amp_y_true: Opt[
            Ant[Union[str, torch.Tensor], 'True source amp. y']
        ] = None,
        src_amp_x_true: Opt[
            Ant[Union[str, torch.Tensor], 'True source amp. x']
        ] = None,
        vp_prmzt: Ant[
            Call[[torch.Tensor], Param], "Parameterized vp"
        ] = Param.delay_init(requires_grad=True),
        vs_prmzt: Ant[
            Call[[torch.Tensor], Param], "Parameterized vs"
        ] = Param.delay_init(requires_grad=False),
        rho_prmzt: Ant[
            Call[[torch.Tensor], Param], "Parameterized rho"
        ] = Param.delay_init(requires_grad=False),
        src_amp_y_prmzt: Ant[
            Call[[torch.Tensor], Param], "Parameterized src amp y"
        ] = Param.delay_init(requires_grad=False),
        src_amp_x_prmzt: Ant[
            Call[[torch.Tensor], Param], "Parameterized src amp x"
        ] = Param.delay_init(requires_grad=False),
    ):
        super().__init__()

        def get(filename, default=None):
            print(f'filename={filename}, path={path}', flush=True)
            if isinstance(filename, torch.Tensor):
                return filename
            elif filename is not None:
                u = get_data3(field=filename, path=path)
                print(f'    shape={u.shape}', flush=True)
                return u
            elif filename is None and default is not None:
                print(
                    f'    Attempt: {path}/{default}.pt...', flush=True, end=''
                )
                if os.path.exists(f'{path}/{default}.pt'):
                    u = get_data3(field=default, path=path)
                else:
                    u = None
                if u is not None:
                    print(f'{u.shape}', flush=True)
                else:
                    print('FAILED...returning None', flush=True)
                return u
            else:
                return None

        def get_prmzt(filename, default=None, *, prmzt):
            tmp = get(filename, default=default)
            if tmp is None:
                return None
            else:
                return prmzt(tmp)

        self.vp = get_prmzt(vp_init, 'vp_init', prmzt=vp_prmzt)
        self.vs = get_prmzt(vs_init, 'vs_init', prmzt=vs_prmzt)
        self.rho = get_prmzt(rho_init, 'rho_init', prmzt=rho_prmzt)

        # self.src_amp_y = torch.nn.Parameter(
        #     get(src_amp_y, 'src_amp_y'), requires_grad=False
        # )
        # self.src_amp_x = torch.nn.Parameter(
        #     get(src_amp_x, 'src_amp_x'), requires_grad=False
        # )

        self.tensors = dict(
            src_amp_y=get(src_amp_y, 'src_amp_y'),
            src_amp_x=get(src_amp_x, 'src_amp_x'),
            src_loc_y=get(src_loc_y, 'src_loc_y'),
            rec_loc_y=get(rec_loc_y, 'rec_loc_y'),
            obs_data=get(obs_data, 'obs_data'),
            vp_true=get(vp_true, 'vp_true'),
            vs_true=get(vs_true, 'vs_true'),
            rho_true=get(rho_true, 'rho_true'),
            src_amp_y_true=get(src_amp_y_true, 'src_amp_y_true'),
            src_amp_x_true=get(src_amp_x_true, 'src_amp_x_true'),
        )

        self.model = 'acoustic' if self.vs is None else 'elastic'
        self.metadata = get_pydict(path, as_class=False)

        self.set_meta_fields()
        self.set_extra_forwards(extra_forward_args)

    def set_extra_forwards(self, extra_forward_args):
        if extra_forward_args is None:
            self.extra_forward_args = {}
        else:
            self.extra_forward_args = extra_forward_args
        if isinstance(self.vp, ParamConstrained):
            maxv = self.vp.custom.maxv
            if isinstance(self.vs, ParamConstrained):
                maxv = min(maxv, self.vs.custom.maxv)
            self.extra_forward_args.update({'max_vel': maxv})

    def set_meta_fields(self):
        custom_dict = {}
        for k, v in self.metadata.items():
            if k in self.__slots__:
                setattr(self, k, v)
            else:
                custom_dict[k] = v
        self.custom = DotDict(custom_dict)

    def chunk_to(
        self, *, rank, world_size, chunk_exclude=None, device_exclude=None
    ):
        chunk_exclude = [] if chunk_exclude is None else chunk_exclude
        device_exclude = [] if device_exclude is None else device_exclude
        new_params = {}
        for name, param in self.named_parameters():
            if name in chunk_exclude:
                new_params[name] = param
            else:
                chunk = torch.chunk(param, world_size)[rank]
                new_params[name] = chunk

        for name, param in self.named_parameters():
            param.data = new_params[name]

        for name, param in self.named_parameters():
            if name not in device_exclude:
                param.data = param.data.to(rank)

        return self

    def report_info(self):
        lines = ['NAME & TYPE & SHAPE & DEVICE & GRAD REQ']
        for name, param in self.named_parameters():
            lines.append(
                ' & '.join(
                    [
                        name,
                        'PARAM',
                        str(param.shape),
                        str(param.device),
                        str(param.requires_grad),
                    ]
                )
            )
        for name, tensor in self.tensors.items():
            lines.append(
                ' & '.join(
                    [
                        name,
                        'TENSOR',
                        str(None if tensor is None else tensor.shape),
                        str(None if tensor is None else tensor.device),
                        'N/A',
                    ]
                )
            )
        printj(lines, extra_space=5, flush=True)

    def param_map(
        self,
        *,
        fn,
        mode,
        suffix,
        exclude=None,
        forward_args=None,
        forward_kwargs=None,
    ):
        exclude = [] if exclude is None else exclude
        forward_args = {} if forward_args is None else forward_args
        forward_kwargs = {} if forward_kwargs is None else forward_kwargs
        suffix = '' if not suffix else f'_{suffix.replace("_", "")}'

        if mode == 'Parameter':
            d = self.named_param_dict()
            rename = {k: f"{k.replace('.p', '')}{suffix}" for k in d.keys()}
        elif mode == 'Tensor':
            d = self.tensors
            rename = {k: f'{k}{suffix}' for k in d.keys()}
        else:
            raise ValueError(
                f'Unknown mode: {mode}, expected Parameter or Tensor'
            )

        for k, v in d.items():
            if k not in exclude:
                d[k] = fn(v, *forward_args, **forward_kwargs)
            elif mode == 'Parameter':
                d[k] = v.data
            else:
                d[k] = v

        res = {}
        for k, v in d.items():
            if k in rename.keys():
                res[rename[k]] = v
            else:
                res[k] = v
        return res

    def full_map(
        self,
        *,
        param_func,
        tensor_func,
        suffix,
        param_exclude=None,
        tensor_exclude=None,
        param_args=None,
        param_kwargs=None,
        tensor_args=None,
        tensor_kwargs=None,
    ):
        noner = lambda x, t: t() if x is None else x
        param_exclude = noner(param_exclude, list)
        tensor_exclude = noner(tensor_exclude, list)
        param_args = noner(param_args, list)
        param_kwargs = noner(param_kwargs, dict)
        tensor_args = noner(tensor_args, list)
        tensor_kwargs = noner(tensor_kwargs, dict)
        suffix = '' if not suffix else f'_{suffix.replace("_", "")}'
        d_param = self.param_map(
            fn=param_func,
            mode='Parameter',
            suffix=suffix,
            exclude=param_exclude,
            forward_args=param_args,
            forward_kwargs=param_kwargs,
        )
        d_tensor = self.param_map(
            fn=tensor_func,
            mode='Tensor',
            suffix=suffix,
            exclude=tensor_exclude,
            forward_args=tensor_args,
            forward_kwargs=tensor_kwargs,
        )
        return d_param, d_tensor

    def chunk2_to(
        self, *, rank, world_size, chunk_exclude=None, device_exclude=None
    ):
        def chunk_param(x):
            if x is None:
                return None
            else:
                return torch.nn.Parameter(torch.chunk(x, world_size)[rank])

        def chunk_tensor(x):
            if x is None:
                return None
            else:
                return torch.chunk(x, world_size)[rank].to(rank)

        d_param, d_tensor = self.full_map(
            param_func=chunk_param,
            tensor_func=chunk_tensor,
            suffix='',
            param_exclude=chunk_exclude,
            tensor_exclude=chunk_exclude,
        )
        for k, v in self.named_parameters():
            v.data = d_param[k.replace('.p', '')].data
        for k, v in self.tensors.items():
            v = d_tensor[k]

        def to_handler(x):
            return x.to(rank) if x is not None else None

        d_param, d_tensor = self.full_map(
            param_func=to_handler,
            tensor_func=to_handler,
            suffix='',
            param_exclude=device_exclude,
            tensor_exclude=device_exclude,
        )
        for k, v in self.named_parameters():
            v.data = d_param[k.replace('.p', '')].data
        for k, v in self.tensors.items():
            v = d_tensor[k]

        return self

    def named_param_dict(self):
        return {k: v for k, v in self.named_parameters()}

    def forward(self, x, **kw):
        kw = {**self.extra_forward_args, **kw}

        if 'amp_idx' in kw.keys():
            amp_idx = kw['amp_idx']
            del kw['amp_idx']
        # else:
        #     amp_idx = torch.arange(self.src_amp_y.shape[0])

        # if( 'amp_idx' in kw.keys() ):
        #     amp_idx = kw['amp_idx']
        #     del kw['amp_idx']
        # else:
        #     amp_idx = torch.arange(self.src_amp_y.shape[0])
        if self.model == 'acoustic':
            # print(f'source_amplitudes.shape={self.src_amp_y.shape}')
            # print(
            #     'source'
            #     f' amplitudes[amp_idx].shape={self.src_amp_y[amp_idx].shape}'
            # )
            # print(f'source_locations.shape={self.src_loc_y.shape}')
            # print(f'receiver_locations.shape={self.rec_loc_y.shape}')
            # print(f'source_amp.device = {self.src_amp_y.device}')
            # print(f'source_loc.device = {self.src_loc_y.device}')
            # print(f'receiver_loc.device = {self.rec_loc_y.device}')
            # print(f'vp.device = {self.vp().device}')
            # print(self.src_amp_y.device, flush=True)
            # print(self.src_amp_y.shape, flush=True)
            return dw.scalar(
                self.vp(),
                4.0,
                0.004,
                source_amplitudes=self.src_amp_y,
                source_locations=self.src_loc_y,
                receiver_locations=self.rec_loc_y,
                pml_freq=25,
                time_pad_frac=0.2,
                max_vel=2500,
            )
        else:
            return dw.elastic(
                *get_lame(self.vp(), self.vs(), self.rho()),
                self.dy,
                self.dt,
                source_amplitudes_y=self.src_amp_y,
                source_locations_y=self.src_loc_y,
                receiver_locations_y=self.rec_loc_y,
                **kw,
            )
