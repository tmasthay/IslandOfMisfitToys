from typing import Callable
import torch
from dataclasses import dataclass
from torch.nn.functional import mse_loss


class W2(torch.nn.Module):
    renorm_func: Callable[[torch.Tensor], torch.Tensor] = lambda x: x
    eps: float = 0.0

    def __post_init__(self):
        super().__init__()

    def prep_data(self, y):
        return self.renorm_func(y + self.eps)

    def forward(self, y_pred, y):
        # y_pred = self.prep_data(y_pred)
        # y = self.prep_data(y)

        return mse_loss(y_pred, y)
