#from Experiments.helpers.import_env import *
from import_env import *
import os

omissions = ['examples', 'Experiments', 'docs']

#run_make_files(omissions)

init_modules(
    os.getcwd(), 
    root=True, 
    omissions=omissions
)


