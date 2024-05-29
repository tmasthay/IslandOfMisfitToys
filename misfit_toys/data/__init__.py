"""
Data generation package that downloads and processes observed synthetic data from given model parameters.
That data can then be read in to benchmark FWI algorithms.
"""

from . import custom, das_curtin, dataset, download_data, marmousi, marmousi2

__all__ = [
    'marmousi2',
    'das_curtin',
    'custom',
    'marmousi',
    'download_data',
    'dataset',
]
