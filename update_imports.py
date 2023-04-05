from Experiments.helpers.import_env import *
import os

init_modules(
    os.getcwd(), 
    root=True, 
    omissions=['examples', 'Experiments']
)


