import pickle
import zlib
import glob
import os

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import numpy as np

from animator import Animator

path = '../checkpoints/main_random_hparams/back_to_original_room_door_and_robots_and_changed_to_sac/2019-12-25-01-56-59-999540'
save = False
start = 38000
num_steps = 1000

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

if save:
    import matplotlib
    matplotlib.use('Agg')

animator = Animator(width=width, height=height, door_width=door_width, radi=[radius] * num_robots, pause_time=None if save else 0.01)

fig = animator.get_figure()

def init_func():
    return animator.circles+animator.dirs

def update(step_loc_dir):
    step = step_loc_dir[0]
    locations = step_loc_dir[1]
    directions = step_loc_dir[2]
    animator.Update(step,locations,directions)
    return animator.circles+animator.dirs

def frames():
    prev_file_ind = None
    history = None
    for step in range(start,start+num_steps):
        file_ind = get_file_ind(step)
        if file_ind != prev_file_ind:
            with open(history_files[file_ind], 'rb') as f:
                history = pickle.loads(zlib.decompress(f.read()))
            prev_file_ind = file_ind
        k = step-start_steps[file_ind]
        yield (step,history[:,k,:2],history[:,k,2])

anim = FuncAnimation(fig,update,frames,init_func,interval=10,repeat=False,blit=False,save_count=num_steps)
if save:
    writer = FFMpegWriter(fps=20, bitrate=120)
    anim.save('vid.mp4', writer=writer)
else:
    plt.show()
