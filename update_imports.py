#from Experiments.helpers.import_env import *
import os
import sys

sys.path.append('%s/helpers/helpers'%os.getcwd())
from import_env import *

include_paths = ['misfits', 'forward']

#run_make_files(omissions)
print('Clearly a change here')

init_modules(
    os.getcwd(), 
    root=True, 
    inclusions=include_paths,
    unload=True
)


