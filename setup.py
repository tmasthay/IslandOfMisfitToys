from setuptools import setup, find_packages

setup(
    name='IslandOfMisfitToys',
    version='0.2.1',
    description='A suite of misfit functions to test FWI',  # Provide a short description here
    author='Tyler Masthay and Yiran Shen',
    author_email='tyler@ices.utexas.edu',
    url='https://github.com/tmasthay/IslandOfMisfitToys',
    packages=find_packages(),
    install_requires=[
        'deepwave>=0.0.19',
        'numpy>=1.19',   # Update the version as needed
        'scipy',
        'pytest',
        'masthay_helpers>=0.2.8',
        'imageio',
        'tqdm',
        'torchsummary'
    ],
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

