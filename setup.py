import os
from subprocess import check_output as co

from setuptools import find_packages, setup

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="IslandOfMisfitToys",
    version="0.5.0",
    description="A suite of misfit functions to test FWI",  # Provide a short description here
    author="Tyler Masthay and Yiran Shen",
    author_email="tyler@ices.utexas.edu",
    url="https://github.com/tmasthay/IslandOfMisfitToys",
    packages=find_packages(),
    install_requires=requirements,
    long_description=open("README.md", "r").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
    ],
    python_requires=">=3.9",
    include_package_data=True,
    zip_safe=False,
    options={"bdist_wheel": {"universal": True}},
)

pip_path = str(co(["which", "pip"]), "utf-8").strip()
os.system(
    f'{pip_path} install'
    ' git+https://github.com/patrick-kidger/torchcubicspline.git'
)
os.system(f'{pip_path} install git+https://github.com/tmasthay/rich_tools.git')
