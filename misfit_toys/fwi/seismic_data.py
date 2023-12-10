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
from deepwave import scalar, elastic
from deepwave.common import vpvsrho_to_lambmubuoyancy as get_lame
from masthay_helpers.global_helpers import DotDict
from misfit_toys.data.dataset import get_data3, get_pydict


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
    for k in remap:
        d[remap[k]] = d.pop(k)
    return d


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
        data[k] = torch.chunk(data[k], world_size)[rank]
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
        vs=None,
        rho=None,
        model='acoustic',
        meta,
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
                self.meta.dx,
                self.meta.dt,
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
