from ..dataset import DataFactoryMeta
from ...utils import DotDict

class Factory(DataFactoryMeta):
    def generate_derived_data(self, *, data):
        d = DotDict(data)
        return d