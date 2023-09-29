from misfit_toys.data.dataset import DataFactory
from rich.traceback import install

install(show_locals=True)

exclusions = ['das_curtin']
DataFactory.create_database(storage='conda/data', exclusions=exclusions)
