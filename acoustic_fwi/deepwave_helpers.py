import os
from subprocess import check_output as co
import sys
from time import time
import matplotlib.pyplot as plt
from imageio import imread, mimsave
import numpy as np
import torch

def sco(s, split=True):
    u = co(s, shell=True).decode('utf-8')
    if( split ): return u.split('\n')[:-1]
    else: return u

def get_file(s, path=''):
    full_path = list(set(path.split(':')) \
        .union(set(sco('echo $CONDA_PREFIX/data'))))
    full_path = [e for e in full_path if e != '']
    if( os.getcwd() not in full_path ):
        full_path.insert(0, os.getcwd())
    for e in full_path:
        if( os.path.exists(e + '/' + s) ): return e + '/' + s
    stars = 80 * '*'
    raise FileNotFoundError('filename "%s" not found in any' + \
        ' of the following directories\n%s\n%s\n%s'%(
            s, stars, '\n'.join(full_path), stars
        )
    )

def run_and_time(start_msg, end_msg, f, *args, **kwargs):
    stars = 80*'*' + '\n'
    print('%s\n%s'%(stars, start_msg), file=sys.stderr)
    start_time = time()
    u = f(*args, **kwargs)
    print('%s ::: %.4f\n%s'%(end_msg, time() - start_time, stars), 
        file=sys.stderr)
    return u

def make_gif(x, folder, the_map='cividis'):
    os.system('mkdir -p %s'%folder)
    for i in range(len(x)):
        plt.imshow(np.transpose(x[i]), cmap=the_map, aspect='auto')
        plt.colorbar()
        plt.title('Epoch %d'%i)
        plt.savefig('%s/%d.jpg'%(folder, i))
        plt.close()
    filenames = ['%s/%d.jpg'%(folder, i) for i in range(len(x))]
    images = [imread(e) for e in filenames]
    mimsave('%s/movie.gif'%folder, images, duration=0.1, loop=0)
    for e in filenames:
        print(e)
        os.system('rm %s'%e)

def report_gpu_memory_allocation(msg, mode=2):
    memory_gb = torch.cuda.memory_allocated() / (1024 ** 3)
    print(f"GPU Memory {msg}: {memory_gb:.2f} GB")

def gpu_mem_helper():
    with open(os.path.expanduser('~/.bash_functions'), 'r') as f:
        lines = f.readlines()

    start_line = next(i for i, line in enumerate(lines) if line.strip() == 'gpu_mem() {')
    end_line = next(i for i, line in enumerate(lines) if i > start_line and line.strip() == '}')

    s = ''.join(lines[start_line+1:end_line])
    def helper(msg=''):
        print('%s...%s'%(msg, ':::'.join(sco(s))))
    return helper



    
