# setup_smart_quantile.py
from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    name = 'smart_quantile',
    ext_modules = cythonize("smart_quantile.pyx"),
    include_dirs=[np.get_include()]
)

