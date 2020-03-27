import pickle
import zlib
import glob
import os
from cycler import cycler

import numpy as np
import matplotlib.pyplot as plt

window_size = 1000

fig, ax = plt.subplots(1)
ax.set_prop_cycle(cycler(linestyle=['-','--',':','-.'])*cycler(color=plt.get_cmap('tab20').colors))

did_title = False
search_path = '../checkpoints/main_random_hparams/hparams1'
for path in glob.glob(os.path.join(search_path ,'*')):

    try:
        with open(os.path.join(path,'params.pickle'),'rb') as f:
            d = pickle.load(f)
    except FileNotFoundError:
        continue

    height = d['height']
    radius = d['radius']
    gamma = d['gamma']
    escape_reward = d['escape_reward']

    history_files = glob.glob(path+'/history_*.pickle')
    if len(history_files) == 0:
        continue
    start_steps = [int(os.path.basename(f).lstrip('history_').rstrip('.pickle')) for f in history_files]
    start_steps,history_files = zip(*sorted(zip(start_steps,history_files)))

    escapes_per_time = []
    rewards_per_robot_per_time = []

    for file in history_files:
        with open(file, 'rb') as f:
            history = pickle.loads(zlib.decompress(f.read()))
            escapes_per_time.append(np.sum(history[:, :, 1] > height - radius, 0))

    escapes_per_time = np.concatenate(escapes_per_time,0)
    escapes_per_window = np.convolve(escapes_per_time,np.ones(window_size),'same')

    plt.plot(escapes_per_window,label=path.lstrip(search_path))

    if not did_title:
        print(('{:<28s} {:<7s} '.format('dir','best') + '{:<22s} '*len(d)).format(*d.keys()))
        did_title = True
    print(('{:<28s} {:<7.1f} '.format(path.lstrip(search_path),np.max(escapes_per_window)) + '{:<22} ' * len(d)).format(*d.values()))
plt.legend()
plt.show()
