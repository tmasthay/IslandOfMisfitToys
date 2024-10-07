from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Union

import torch
import torch.nn.functional as F
from dotmap import DotMap


def gamma(x: torch.Tensor):
    if isinstance(x, int) or isinstance(x, float):
        x = torch.Tensor([float(x)])
    return torch.exp(torch.special.gammaln(x))


class DiffFunction(ABC):
    @abstractmethod
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def deriv(self, x: torch.Tensor, *, alpha: float) -> torch.Tensor:
        raise NotImplementedError("deriv must be overridden by subclass")


@dataclass(kw_only=True)
class NonAnalyticFunction(DiffFunction):
    @abstractmethod
    def _class_deriv(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def deriv(self, x: torch.Tensor, *, alpha: float) -> torch.Tensor:
        res = torch.nan * torch.ones(alpha.shape[0], x.shape[0])
        for i, e in enumerate(alpha):
            if e == 0.0:
                res[i] = self.__call__(x)
            elif e == 1.0:
                res[i] = self._class_deriv(x)
        return res


@dataclass(kw_only=True)
class Power(DiffFunction):
    beta: float
    scale: float = 1.0

    # def __post_init__(self):
    #     self.beta = torch.tensor(self.beta)
    #     self.scale = torch.tensor(self.scale)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * x**self.beta

    def deriv(self, x: torch.Tensor, *, alpha: torch.Tensor) -> torch.Tensor:
        diff = self.beta - alpha[:, None]
        prefactor = (
            self.scale
            * gamma(torch.tensor(self.beta + 1.0))
            / gamma(diff + 1.0)
        )
        return prefactor * x**diff


@dataclass(kw_only=True)
class Sine(NonAnalyticFunction):
    omega: float = 1.0
    scale: float = 1.0
    phase: float = 0.0

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * torch.sin(self.omega * x + self.phase)

    def _class_deriv(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * self.omega * torch.cos(self.omega * x + self.phase)

        # return res


@dataclass(kw_only=True)
class SineConfident(DiffFunction):
    omega: float = 1.0
    scale: float = 1.0
    phase: float = 0.0

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * torch.sin(self.omega * x + self.phase)

    def deriv(self, x: torch.Tensor, *, alpha: torch.Tensor) -> torch.Tensor:
        return self.omega ** alpha[:, None] * torch.sin(
            self.omega * x[None, :] + self.phase + torch.pi * alpha[:, None] / 2
        )


@dataclass(kw_only=True)
class DataFunction(NonAnalyticFunction):
    path: str
    device: str = 'cpu'
    __slice__: list[Union[list[str], None]] = None
    squeeze: bool = True

    def __post_init__(self):
        if self.__slice__ is None:
            self.slyce = slice(None)
        else:
            self.slyce = []
            for i, e in enumerate(self.__slice__):
                if e is None:
                    self.slyce.append(slice(None))
                elif isinstance(e, list):
                    self.slyce.append(slice(*e))
                elif isinstance(e, int):
                    self.slyce.append(slice(e, e + 1))
                else:
                    raise ValueError(f'Invalid slice: {e}')
        # input(self.slyce)
        self.data = torch.load(self.path)[self.slyce].to(self.device)
        if self.squeeze:
            self.data = self.data.squeeze()
        # input(self.data.shape)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.data

    def _class_deriv(self, x: torch.Tensor) -> torch.Tensor:
        dx = x[1] - x[0]
        u = (self.data[1:] - self.data[:-1]) / dx
        return F.pad(u, (1, 0), value=u[0].item())


@dataclass(kw_only=True)
class FourierSineSeries(DiffFunction):
    # omega: torch.Tensor = None
    # scale: torch.Tensor = None
    # phase: torch.Tensor = None
    basis_terms: Union[torch.Tensor, list[list[float]]]
    device: list[str] = 'cpu'

    def __post_init__(self):
        if self.basis_terms is None:
            self._basis = torch.Tensor([[1.0, 1.0, 0.0]])
        elif isinstance(self.basis_terms, list):
            self._basis = torch.Tensor(self.basis_terms)
        elif isinstance(self.basis_terms, torch.Tensor):
            self._basis = self.basis_terms
        else:
            raise ValueError(f'Invalid basis_terms: {self.basis_terms}')
        self._basis = self._basis.to(self.device)
        self.scale = self._basis[:, 0]
        self.omega = self._basis[:, 1]
        self.phase = self._basis[:, 2]

    def broadcast(
        self, *, scale=None, omega=None, phase=None, alpha=None, x=None
    ):
        scale_exp = None if scale is None else scale[:, None, None]
        omega_exp = None if omega is None else omega[:, None, None]
        phase_exp = None if phase is None else phase[:, None, None]
        alpha_exp = None if alpha is None else alpha[None, :, None]
        x_exp = None if x is None else x[None, None, :]
        d = DotMap(
            scale=scale_exp,
            omega=omega_exp,
            phase=phase_exp,
            alpha=alpha_exp,
            x=x_exp,
        )
        return d

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # return self.scale * torch.sin(self.omega * x + self.phase)
        # return torch.sum(self.scale[:, None] * torch.sin(self.omega[:, None] * x[None, :] + self.phase[:, None]), dim=0)
        broad = self.broadcast(
            scale=self.scale,
            omega=self.omega,
            phase=self.phase,
            alpha=None,
            x=x,
        )
        all_terms = broad.scale * torch.sin(broad.omega * x + broad.phase)
        return torch.sum(all_terms, dim=0).squeeze()

    def deriv(self, x: torch.Tensor, *, alpha: torch.Tensor) -> torch.Tensor:
        broad = self.broadcast(
            scale=self.scale,
            omega=self.omega,
            phase=self.phase,
            alpha=alpha,
            x=x,
        )
        scale_term = broad.scale * broad.omega**broad.alpha
        freq_term = (
            broad.omega * broad.x + broad.phase + torch.pi * broad.alpha / 2
        )
        non_summed_term = scale_term * torch.sin(freq_term)
        return torch.sum(non_summed_term, dim=0)


def get_callback(s: str, **kw):
    d = {
        'power': Power,
        'sine': Sine,
        'sine_confident': SineConfident,
        'data': DataFunction,
        'fourier': FourierSineSeries,
    }
    return d[s](**kw)
