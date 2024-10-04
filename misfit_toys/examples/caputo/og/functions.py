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
        raise NotImplementedError(
            "Need to supply the derivative if you want to use it"
        )

@dataclass(kw_only=True)
class Power(DiffFunction):
    beta: float
    scale: float = 1.0

    def __post_init__(self):
        self.beta = torch.tensor(self.beta)
        self.scale = torch.tensor(self.scale)
        
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * x**self.beta

    def deriv(self, x: torch.Tensor, *, alpha: float) -> torch.Tensor:
        prefactor = (
            self.scale
            * gamma(self.beta + 1.0)
            / gamma(self.beta - alpha + 1.0)
        )
        return prefactor * x ** (self.beta - alpha)


def get_callback(s: str, **kw):
    d = {'power': Power}
    return d[s](**kw)
