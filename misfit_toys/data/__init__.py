"""
Data generation package that downloads and processes observed synthetic data from given model parameters.
That data can then be read in to benchmark FWI algorithms.

Usage:
    Navigate to the data directory and run the download_data script with the desired folders::

        cd misfit_toys/data
        python download_data [folder1] [folder2] ...

This will download and process the data for use in benchmarking FWI algorithms.
"""

from . import custom, das_curtin, dataset, download_data, marmousi, marmousi2

__all__ = [
    'marmousi',
    'marmousi2',
    'das_curtin',
    'custom',
    'dataset',
    'download_data',
]
