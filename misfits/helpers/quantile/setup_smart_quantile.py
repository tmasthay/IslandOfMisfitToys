# setup22.py
from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules = cythonize("smart_quantile_cython.pyx"),
    include_dirs=[np.get_include()]
)

