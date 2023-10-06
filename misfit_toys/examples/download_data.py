import misfit_toys
from misfit_toys.data.dataset import DataFactory
from masthay_helpers.global_helpers import iprint


def download_data():
    storage = 'conda/data'
    exclusions = ['das_curtin']
    DataFactory.create_database(storage=storage, exclusions=exclusions)

    iprint(
        (
            'To see more examples, look at the source code located in'
            f' {__file__}.\n'
        ),
        'You can still access those examples directly through misfit_toys.',
        'examples if you do not need the actual source code.',
    )