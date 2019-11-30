from itertools import count

from agent import Robot
from animator import Animator
from environment import Room
from policies import HeuristicPolicy, DQNPolicy
from utils import ReplayMemory

with_animation = True
view_interval = 1000
view_duration = 50

room = Room(
    width=14,
    height=22,
    door_width=6,
    max_episode_steps=3000
)
room.populate_with_robots(num_robots=1)

replay_buffer = ReplayMemory(capacity=10000)
dqn_policy = DQNPolicy(
    state_shape=(Robot.num_fov_pixels, 2, 4),
    num_actions=3 * 3,
    widths=[128, ],
    replay_buffer_to=replay_buffer,
    eps_start=0.9,
    eps_end=0.05,
    eps_decay=50000,
    batch_size=32,
    gamma=0.99,
    optimization_interval=5,
    target_update_interval=1000,
    lr=1e-3,
    optimizer='ADAM'
)
heuristic_policy = HeuristicPolicy(replay_buffer)

room.assign_policies([dqn_policy])

if with_animation:
    animator = Animator(room, robots_to_debug=[room.robots[0]])
    animator.Update()

for step in count():
    display = with_animation and ((step % view_interval) < view_duration)
    room.global_step(inference=False)  # inference=display
    if display:
        animator.Update()
