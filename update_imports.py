# from Experiments.helpers.import_env import *
import os
import sys

from mh.import_env import init_modules

include_paths = ["misfit_toys"]
omissions = ['multirun', 'outputs']

# run_make_files(omissions)

init_modules(
    os.getcwd(),
    root=True,
    inclusions=include_paths,
    unload=True,
    omissions=omissions,
)
