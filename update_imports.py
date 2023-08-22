#from Experiments.helpers.import_env import *
import os
import sys
from import_env import *

include_paths = ['misfits', 'acoustic_fwi']

#run_make_files(omissions)

init_modules(
    os.getcwd(), 
    root=True, 
    inclusions=include_paths,
    unload=True
)


