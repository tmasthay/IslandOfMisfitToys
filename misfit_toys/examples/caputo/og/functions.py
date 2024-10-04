from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch

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
        return prefactor * x ** diff
    
@dataclass(kw_only=True)
class Sine(NonAnalyticFunction):
    omega: float = 1.0
    scale: float = 1.0
    phase: float = 0.0
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * torch.sin(self.omega * x + self.phase)
    
    def _class_deriv(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * self.omega * torch.cos(self.omega * x + self.phase)
    
        return res

@dataclass(kw_only=True)
class SineConfident(DiffFunction):
    omega: float = 1.0
    scale: float = 1.0
    phase: float = 0.0
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * torch.sin(self.omega * x + self.phase)

    def deriv(self, x: torch.Tensor, *, alpha: torch.Tensor) -> torch.Tensor:
        return self.omega ** alpha[:, None] * torch.sin(self.omega * x[None, :] + self.phase + torch.pi * alpha[:, None] / 2)
    
def get_callback(s: str, **kw):
    d = {'power': Power, 'sine': Sine, 'sine_confident': SineConfident}
    return d[s](**kw)
