from ..dataset import *
from ...utils import DotDict
from .metadata import metadata
import os
import torch
from warnings import warn
import deepwave as dw 

class Factory(DataFactory):
    def __init__(self, *, path, device=None):
        super().__init__(path=path)
        if( device is None ):
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        self.metadata = metadata()

    def generate_derived_data(self, *, data):
        pass

    def manufacture_data(self):
        self._manufacture_data(metadata=self.metadata)