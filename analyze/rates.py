import pickle
import zlib
import glob
import os

import numpy as np
import matplotlib.pyplot as plt

path = '../checkpoints/main/2019-12-25-07-35-55-054944/'
window_size = 1000

with open(os.path.join(path,'params.pickle'),'rb') as f:
    d = pickle.load(f)
print(d)

height = d['height']
radius = d['radius']
gamma = d['gamma']
escape_reward = d['escape_reward']



history_files = glob.glob(path+'/history_*.pickle')
start_steps = [int(os.path.basename(f).lstrip('history_').rstrip('.pickle')) for f in history_files]
start_steps,history_files = zip(*sorted(zip(start_steps,history_files)))

escapes_per_time = []
rewards_per_robot_per_time = []

for file in history_files:
    with open(file, 'rb') as f:
        history = pickle.loads(zlib.decompress(f.read()))
        escapes_per_time.append(np.sum(history[:, :, 1] > height - radius, 0))
        rewards_per_robot_per_time.append(history[:, :, 3])

escapes_per_time = np.concatenate(escapes_per_time,0)
rewards_per_robot_per_time = np.concatenate(rewards_per_robot_per_time,1)

escapes_per_window = np.convolve(escapes_per_time,np.ones(window_size),'same')

total_returns = []
episode_end = []
episode_duration = []
for rewards in rewards_per_robot_per_time:
    episode_end_inds = np.nonzero(rewards==escape_reward)[0]+1
    episode_start_inds = np.concatenate(([0],episode_end_inds))
    for start,stop in zip(episode_start_inds,episode_end_inds):
        r = rewards[start:stop]
        discount = np.power(gamma,np.arange(len(r)))
        total_returns.append(np.sum(r*discount))
        episode_end.append(stop-1)
        episode_duration.append(len(r))

total_returns = np.array(total_returns)
episode_end = np.array(episode_end)
episode_duration = np.array(episode_duration)
_,inds = np.unique(episode_end, return_index=True)
episode_end = episode_end[inds]
total_returns  = total_returns[inds]
episode_duration = episode_duration[inds]
inds = np.argsort(episode_end)
episode_end = episode_end[inds]
total_returns = total_returns[inds]
episode_duration = episode_duration[inds]

returns_in_window = []
durations_in_window = []
for j,ee in enumerate(episode_end):
    i = np.argmax(episode_end > ee-window_size)
    returns_in_window.append(np.mean(total_returns[i:(j+1)]))
    durations_in_window.append(np.mean(episode_duration[i:(j+1)]))


plt.figure()
# plt.plot(episode_end, total_returns)
plt.plot(episode_end, returns_in_window)
plt.title('returns_in_window')

plt.figure()
# plt.plot(episode_end, total_returns)
plt.plot(episode_end, durations_in_window)
plt.title('durations_in_window')

plt.figure()
plt.plot(escapes_per_window)
plt.title('escapes_per_window')

plt.show()
