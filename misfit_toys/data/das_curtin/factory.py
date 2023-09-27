from ..dataset import DataFactory
from ...utils import DotDict


class Factory(DataFactory):
    def _manufacture_data(self):
        d = DotDict(self.process_web_data())
        return d
