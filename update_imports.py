#from Experiments.helpers.import_env import *
from import_env import *
import os

include_paths = ['misfits', 'forward']

#run_make_files(omissions)

input(os.getcwd())

init_modules(
    os.getcwd(), 
    root=True, 
    omissions=omissions,
    unload=True
)


