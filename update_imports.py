from Experiments.helpers.import_env import *
import os

run_make_files()

init_modules(
    os.getcwd(), 
    root=True, 
    omissions=['examples', 'Experiments']
)


