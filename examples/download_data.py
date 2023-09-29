from misfit_toys.data.dataset import DataFactory
from rich.traceback import install

install(show_locals=True)

storage = 'conda/data'
exclusions = ['das_curtin']
DataFactory.create_database(storage=storage, exclusions=exclusions)
