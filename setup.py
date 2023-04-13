from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext as _build_ext
from Cython.Build import cythonize
import numpy as np

class build_ext(_build_ext):
    def finalize_options(self):
        _build_ext.finalize_options(self)
        # Prevent numpy from being imported before it's installed.
        __builtins__.__NUMPY_SETUP__ = False
        self.include_dirs.append(np.get_include())

extensions = [
    Extension(
        'misfits.helpers.quantile.smart_quantile_cython',
        ['misfits/helpers/quantile/smart_quantile_cython.pyx'],
        include_dirs=[np.get_include()],
    )
]

setup(
    name='IslandOfMisfitToys',
    version='0.1.0',
    description='A suite of misfit functions to test FWI',  # Provide a short description here
    author='Tyler Masthay and Yiran Shen',
    author_email='tyler@ices.utexas.edu',
    url='https://github.com/tmasthay/IslandOfMisfitToys',
    packages=find_packages(),
    ext_modules=cythonize(extensions),
    cmdclass={'build_ext': build_ext},
    install_requires=[
        'Cython>=0.29',  # Update the version as needed
        'numpy>=1.19',   # Update the version as needed
        # Add other dependencies here
    ],
)

