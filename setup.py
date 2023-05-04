from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext as _build_ext
import os

import os

def check_and_install_dependencies(dependencies):
    for dependency in dependencies:
        try:
            __import__(dependency)
        except ImportError:
            print(f"{dependency} not found. Installing...")
            os.system(f"pip install {dependency}")

names = ['cython', 'numpy']
check_and_install_dependencies(names)

for name in names:
    os.system('pip show %s > tmp_%s_check.txt'%(name.replace('c','C'),name))
    not_found = 'WARNING: Package(s) not found: %s'%name.replace('c','C')
    with open('tmp_%s_check.txt'%name,'r') as f1:
        if( not_found  in f1.read() ):
            os.system('pip install %s'%name.replace('c','C'))

from Cython.Build import cythonize
import numpy as np

#print('Pypi CI test')

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
    package_data={'misfits': ['helpers/quantile/*.pyx', 'helpers/quantile/*.py']},
    ext_modules=cythonize(extensions),
    cmdclass={'build_ext': build_ext},
    install_requires=[
        'Cython>=0.29',  # Update the version as needed
        'numpy>=1.19',   # Update the version as needed
        'scipy',
        'pytest'
        # Add other dependencies here
    ],
    long_description=open('README.md','r').read(),
    long_description_content_type='text/markdown'
)

