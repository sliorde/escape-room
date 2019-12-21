import pickle
import zlib
import glob
import os

import numpy as np

from animator import Animator

path = '../checkpoints/main/2019-12-20-21-33-36-625028'
start = 168000
num_steps = 3000

with open(os.path.join(path,'params.pickle'),'rb') as f:
    d = pickle.load(f)

height = d['height']
width = d['width']
door_width = d['door_width']
radius = d['radius']

history_files = glob.glob(path+'/history_*.pickle')
start_steps = [int(os.path.basename(f).lstrip('history_').rstrip('.pickle')) for f in history_files]
start_steps,history_files = zip(*sorted(zip(start_steps,history_files)))

start_steps = np.array(start_steps)

def get_file_ind(step):
    return np.argmax(step < start_steps)-1

with open(history_files[0], 'rb') as f:
    num_robots = pickle.loads(zlib.decompress(f.read())).shape[0]

animator = Animator(width=width, height=height, door_width=door_width, radi=[radius] * num_robots, pause_time=0.01)

prev_file_ind = None
history = None
for step in range(start, start + num_steps):
    file_ind = get_file_ind(step)
    if file_ind != prev_file_ind:
        with open(history_files[file_ind], 'rb') as f:
            history = pickle.loads(zlib.decompress(f.read()))
        prev_file_ind = file_ind
    k = step - start_steps[file_ind]
    animator.Update(step, history[:, k, :2], history[:, k, 2])