# setup_smart_quantile.py
from setuptools import setup
from Cython.Build import cythonize
import numpy as np
import os

extensions = cythonize('smart_quantile.pyx')

build_folder = '../../../build'
if( not os.path.exists(build_folder) ): 
    os.system('mkdir %s'%build_folder)

for ext in extensions:
    ext.build_temp = build_folder
    ext.build_lib = build_folder

setup(
    name='smart_quantile',
    ext_modules=extensions,
    include_dirs=[np.get_include()]
)

