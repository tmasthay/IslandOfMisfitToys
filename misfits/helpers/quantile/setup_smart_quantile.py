# setup_smart_quantile.py
from setuptools import setup
from Cython.Build import cythonize
import numpy as np
import os

extensions = cythonize('smart_quantile.pyx')

setup(
    name='smart_quantile',
    ext_modules=extensions,
    include_dirs=[np.get_include()]
)

