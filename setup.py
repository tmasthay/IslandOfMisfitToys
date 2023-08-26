from setuptools import setup, find_packages
import os

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name='IslandOfMisfitToys',
    version='0.2.10',
    description='A suite of misfit functions to test FWI',  # Provide a short description here
    author='Tyler Masthay and Yiran Shen',
    author_email='tyler@ices.utexas.edu',
    url='https://github.com/tmasthay/IslandOfMisfitToys',
    packages=find_packages(),
    install_requires=requirements,
    long_description=open('README.md','r').read(),
    long_description_content_type='text/markdown',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering'
    ],
    python_requires='>=3.9',
    include_package_data=True,
    zip_safe=False,
    options={'bdist_wheel': {'universal': True}}
)

