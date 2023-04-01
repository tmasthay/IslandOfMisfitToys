from import_env import *
import os

u = get_subfolders(os.getcwd())
u = [e for e in u if e != 'examples']
for e in u:
    init_modules(e)


