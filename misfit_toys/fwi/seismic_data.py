"""
This module contains classes and functions related to seismic data processing and modeling.

Classes:
    Param: A class representing a parameter.
    ParamConstrained: A class representing a constrained parameter.
    SeismicProp: A class representing a seismic propagation model.

Functions:
    path_builder: Builds a dictionary of seismic data given a path and optional remapping.
    chunk_params: Chunks the parameters based on the rank and world size.
    chunk_tensors: Chunks the tensors based on the rank and world size.
    deploy_data: Deploys the data to the specified rank.
    chunk_and_deploy: Chunks and deploys the data based on the rank and world size.

"""

import torch
from deepwave import elastic, scalar
from deepwave.common import vpvsrho_to_lambmubuoyancy as get_lame
from mh.core import DotDict

from misfit_toys.data.dataset import get_data3, get_pydict
from misfit_toys.utils import tensor_summary


def path_builder(path, *, remap=None, **kw):
    """
    Builds a dictionary of seismic data given a path and optional remapping.

    Args:
        path (str): The path to the seismic data.
        remap (dict, optional): A dictionary specifying the remapping of keys. Defaults to None.
        **kw: Additional keyword arguments representing the fields of the seismic data.

    Returns:
        dict: A dictionary containing the seismic data.

    """
    remap = remap or {}
    d = {}
    for k, v in kw.items():
        if v is None:
            d[k] = get_data3(field=k, path=path)
        else:
            d[k] = v(get_data3(field=k, path=path))
    d['meta'] = get_pydict(path=path, as_class=True)
    for k in remap.keys():
        d[remap[k]] = d.pop(k)
    return d


class Param(torch.nn.Module):
    """
    A class representing a parameter.

    Args:
        p (torch.Tensor): The parameter tensor.
        requires_grad (bool, optional): Whether the parameter requires gradient computation. Defaults to False.
        **kw: Additional keyword arguments.

    Attributes:
        p (torch.nn.Parameter): The parameter tensor.
        custom (DotDict): A dictionary containing custom attributes.

    """

    def __init__(self, *, p, requires_grad=False, **kw):
        super().__init__()
        self.p = torch.nn.Parameter(p, requires_grad=requires_grad)
        self.custom = DotDict(kw)

    def chunk(self, rank, world_size):
        """
        Chunks the parameter based on the rank and world size.

        Args:
            rank (int): The rank of the current process.
            world_size (int): The total number of processes.

        """
        self.p.data = torch.chunk(self.p.data, world_size)[rank]

    def forward(self):
        """
        Returns the parameter tensor.

        Returns:
            torch.Tensor: The parameter tensor.

        """
        return self.p

    @classmethod
    def delay_init(cls, **kw):
        """
        Delays the initialization of the parameter.

        Args:
            **kw: Additional keyword arguments.

        Returns:
            function: A function that initializes the parameter.

        """
        return lambda p: cls(p=p, **kw)

    @classmethod
    def clone(cls, obj, *, requires_grad=None, **kw):
        """
        Clones the parameter.

        Args:
            p (torch.Tensor): The parameter tensor.
            **kw: Additional keyword arguments.

        Returns:
            Param: The cloned parameter.

        """
        requires_grad = (
            obj.p.requires_grad if requires_grad is None else requires_grad
        )
        return cls(p=obj.p.clone(), requires_grad=requires_grad, **kw)


class ParamConstrained(Param):
    """
    A class representing a constrained parameter.

    Args:
        p (torch.Tensor): The parameter tensor.
        minv (float): The minimum value of the parameter.
        maxv (float): The maximum value of the parameter.
        requires_grad (bool, optional): Whether the parameter requires gradient computation. Defaults to False.

    """

    def __init__(self, *, p, minv, maxv, requires_grad=False):
        super().__init__(
            p=torch.logit((p - minv) / (maxv - minv)),
            requires_grad=requires_grad,
            minv=minv,
            maxv=maxv,
        )
        if torch.isnan(self.p).any():
            msg = f'Failed to initialize ParamConstrained with minv={minv}, maxv={maxv}'
            passed_min, passed_max = p.min().item(), p.max().item()
            msg += f'\nTrue max/min of passed data:\n    min={passed_min}, max={passed_max}'
            msg += f'\nConstrained max/min passed into constructor:\n    min={minv}, max={maxv}'
            raise ValueError(msg)

    def forward(self):
        """
        Returns the constrained parameter tensor.

        Returns:
            torch.Tensor: The constrained parameter tensor.

        """
        minv = self.custom.minv
        maxv = self.custom.maxv
        return torch.sigmoid(self.p) * (maxv - minv) + minv


class SeismicPropLegacy(torch.nn.Module):
    """
    A class representing a legacy seismic propagation model.

    Args:
        vp (torch.Tensor): The P-wave velocity tensor.
        vs (torch.Tensor, optional): The S-wave velocity tensor. Defaults to None.
        rho (torch.Tensor, optional): The density tensor. Defaults to None.
        model (str, optional): The type of model ('acoustic' or 'elastic'). Defaults to 'acoustic'.
        dx (float): The spatial step size.
        dt (float): The temporal step size.
        src_amp_y (torch.Tensor, optional): The source amplitudes in the y-direction. Defaults to None.
        src_loc_y (torch.Tensor, optional): The source locations in the y-direction. Defaults to None.
        rec_loc_y (torch.Tensor, optional): The receiver locations in the y-direction. Defaults to None.
        src_amp_x (torch.Tensor, optional): The source amplitudes in the x-direction. Defaults to None.
        src_loc_x (torch.Tensor, optional): The source locations in the x-direction. Defaults to None.
        rec_loc_x (torch.Tensor, optional): The receiver locations in the x-direction. Defaults to None.
        **kw: Additional keyword arguments.

    """

    def __init__(
        self,
        *,
        vp,
        vs=None,
        rho=None,
        model='acoustic',
        dx,
        dt,
        src_amp_y=None,
        src_loc_y=None,
        rec_loc_y=None,
        src_amp_x=None,
        src_loc_x=None,
        rec_loc_x=None,
        **kw,
    ):
        super().__init__()
        self.vp = vp
        self.vs = vs
        self.rho = rho
        self.model = model.lower()
        self.dx = dx
        self.dt = dt
        self.src_amp_y = src_amp_y
        self.src_loc_y = src_loc_y
        self.rec_loc_y = rec_loc_y
        self.src_amp_x = src_amp_x
        self.src_loc_x = src_loc_x
        self.rec_loc_x = rec_loc_x
        self.extra_forward = kw
        self.__validate_init__()

    def __validate_init__(self):
        """
        Validates the initialization of the seismic propagation model.

        Raises:
            ValueError: If the model is not 'acoustic' or 'elastic'.
            ValueError: If the required parameters are not set.

        """
        if self.model not in ['acoustic', 'elastic']:
            raise ValueError(
                f"model must be 'acoustic' or 'elastic', not {self.model}"
            )
        y_set = not any(
            [
                e is None
                for e in [self.src_amp_y, self.rec_loc_y, self.src_loc_y]
            ]
        )
        x_set = not any(
            [
                e is None
                for e in [self.src_amp_x, self.rec_loc_x, self.src_loc_x]
            ]
        )
        if not (y_set or x_set) or ((not y_set) and self.model == 'acoustic'):
            raise ValueError(
                'acoustic model requires y set, elastic y or x set'
            )

    def __get_optional_param__(self, name):
        """
        Returns the optional parameter if it is set, otherwise returns None.

        Args:
            name (str): The name of the optional parameter.

        Returns:
            torch.Tensor or None: The optional parameter tensor or None.

        """
        if getattr(self, name) is None:
            return None
        else:
            return getattr(self, name)()

    def forward(self, dummy):
        """
        Performs forward propagation based on the model type.

        Args:
            dummy: A dummy input.

        Returns:
            torch.Tensor: The output tensor.

        """
        if self.model.lower() == 'acoustic':
            return scalar(
                self.vp(),
                self.dx,
                self.dt,
                source_amplitudes=self.src_amp_y(),
                source_locations=self.src_loc_y,
                receiver_locations=self.rec_loc_y,
                **self.extra_forward,
            )
        elif self.model.lower() == 'elastic':
            lame_params = get_lame(self.vp(), self.vs(), self.rho())
            return elastic(
                *lame_params,
                self.dx,
                self.dt,
                source_amplitudes_y=self.__get_optional_param__('src_amp_y'),
                source_locations_y=self.__get_optional_param__('src_loc_y'),
                receiver_locations_y=self.__get_optional_param__('rec_loc_y'),
                source_amplitudes_x=self.__get_optional_param__('src_amp_x'),
                source_locations_x=self.__get_optional_param__('src_loc_x'),
                receiver_locations_x=self.__get_optional_param__('rec_loc_x'),
                **self.extra_forward,
            )


class SeismicProp(torch.nn.Module):
    """
    A class representing a seismic propagation model.

    Args:
        vp (torch.Tensor): The P-wave velocity tensor.
        vs (torch.Tensor, optional): The S-wave velocity tensor. Defaults to None.
        rho (torch.Tensor, optional): The density tensor. Defaults to None.
        model (str, optional): The type of model ('acoustic' or 'elastic'). Defaults to 'acoustic'.
        meta: The metadata of the seismic data.
        src_amp_y (torch.Tensor, optional): The source amplitudes in the y-direction. Defaults to None.
        src_loc_y (torch.Tensor, optional): The source locations in the y-direction. Defaults to None.
        rec_loc_y (torch.Tensor, optional): The receiver locations in the y-direction. Defaults to None.
        src_amp_x (torch.Tensor, optional): The source amplitudes in the x-direction. Defaults to None.
        src_loc_x (torch.Tensor, optional): The source locations in the x-direction. Defaults to None.
        rec_loc_x (torch.Tensor, optional): The receiver locations in the x-direction. Defaults to None.
        **kw: Additional keyword arguments.

    """

    def __init__(
        self,
        *,
        vp,
        meta,
        vs=None,
        rho=None,
        src_amp_y=None,
        src_loc_y=None,
        rec_loc_y=None,
        src_amp_x=None,
        src_loc_x=None,
        rec_loc_x=None,
        **kw,
    ):
        super().__init__()
        self.__check_nan_inf__(vp if vp is None else vp.p, 'vp')
        self.__check_nan_inf__(vs if vs is None else vs.p, 'vs')
        self.__check_nan_inf__(rho if rho is None else rho.p, 'rho')
        self.vp = vp
        self.vs = vs
        self.rho = rho
        self.model = 'acoustic' if vs is None else 'elastic'
        self.meta = meta
        self.src_amp_y = src_amp_y
        self.src_loc_y = src_loc_y
        self.rec_loc_y = rec_loc_y
        self.src_amp_x = src_amp_x
        self.src_loc_x = src_loc_x
        self.rec_loc_x = rec_loc_x
        self.extra_forward = kw
        self.__validate_init__()

    def __validate_init__(self):
        """
        Validates the initialization of the seismic propagation model.

        Raises:
            ValueError: If the model is not 'acoustic' or 'elastic'.
            ValueError: If the required parameters are not set.

        """
        if self.model not in ['acoustic', 'elastic']:
            raise ValueError(
                f"model must be 'acoustic' or 'elastic', not {self.model}"
            )
        y_set = not any(
            [
                e is None
                for e in [self.src_amp_y, self.rec_loc_y, self.src_loc_y]
            ]
        )
        x_set = not any(
            [
                e is None
                for e in [self.src_amp_x, self.rec_loc_x, self.src_loc_x]
            ]
        )
        if not (y_set or x_set) or ((not y_set) and self.model == 'acoustic'):
            raise ValueError(
                'acoustic model requires y set, elastic y or x set'
            )

    def __get_optional_param__(self, name):
        """
        Returns the optional parameter if it is set, otherwise returns None.

        Args:
            name (str): The name of the optional parameter.

        Returns:
            torch.Tensor or None: The optional parameter tensor or None.

        """
        if getattr(self, name) is None:
            return None
        else:
            return getattr(self, name)()

    def __check_nan_inf__(self, tensor, name=''):
        if tensor is None:
            return
        num_nans = torch.isnan(tensor).sum().item()
        num_infs = torch.isinf(tensor).sum().item()
        prop_nans = num_nans / tensor.numel()
        prop_infs = num_infs / tensor.numel()
        if num_nans > 0 or num_infs > 0:
            msg = f'num_nans={num_nans}, prop_nans={prop_nans}'
            msg += f'\nnum_infs={num_infs}, prop_infs={prop_infs}'
            raise ValueError(f'Tensor {name} invalid: {msg}')

    def forward(self, s):
        """
        Performs forward propagation based on the model type.

        Args:
            dummy: A dummy input.

        Returns:
            torch.Tensor: The output tensor.

        """
        s = s if s is not None else slice(None)
        if self.model.lower() == 'acoustic':
            if torch.isnan(self.vp.p).any():
                raise ValueError(
                    f'Invalid vp before composition due to nan: {tensor_summary(self.vp.p)}'
                )
            v = self.vp()
            if torch.isnan(v).any() or torch.isinf(v).any():
                msg = f'Invalid vp due to nan or inf: {tensor_summary(v)}'
                num_nans = torch.isnan(v).sum().item()
                prop_nans = num_nans / v.numel()
                num_infs = torch.isinf(v).sum().item()
                prop_infs = num_infs / v.numel()
                msg += f'\nnum_nans={num_nans}, prop_nans={prop_nans}'
                msg += f'\nnum_infs={num_infs}, prop_infs={prop_infs}'
                raise ValueError(msg)

            try:
                return scalar(
                    self.vp(),
                    self.meta.dx,
                    self.meta.dt,
                    source_amplitudes=self.src_amp_y()[s],
                    source_locations=self.src_loc_y[s],
                    receiver_locations=self.rec_loc_y[s],
                    **self.extra_forward,
                )
            except Exception as e:
                raise ValueError(
                    f'Original error: {e}\n'
                    f'\nsrc_loc_y.shape={self.src_loc_y.shape}',
                    f'\nsrc_amp_y.shape={self.src_amp_y().shape}',
                    f'\nrec_loc_y.shape={self.rec_loc_y.shape}',
                    f'\ns={s}',
                )
        elif self.model.lower() == 'elastic':
            lame_params = get_lame(self.vp(), self.vs(), self.rho())
            src_amp_y = self.__get_optional_param__('src_amp_y')
            src_loc_y = self.__get_optional_param__('src_loc_y')
            rec_loc_y = self.__get_optional_param__('rec_loc_y')
            src_amp_x = self.__get_optional_param__('src_amp_x')
            src_loc_x = self.__get_optional_param__('src_loc_x')
            rec_loc_x = self.__get_optional_param__('rec_loc_x')
            return elastic(
                *lame_params,
                self.meta.dx,
                self.meta.dt,
                source_amplitudes_y=src_amp_y[s],
                source_locations_y=src_loc_y[s],
                receiver_locations_y=rec_loc_y[s],
                source_amplitudes_x=src_amp_x[s],
                source_locations_x=src_loc_x[s],
                receiver_locations_x=rec_loc_x[s],
                **self.extra_forward,
            )
