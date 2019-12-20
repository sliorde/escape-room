from itertools import count

from agent import Robot
from environment import Room
from rewards import Rewards
from policies import HeuristicPolicy, DQNPolicy
from utils import PrioritizedReplayBuffer
from utils import get_output_dir, save_to_zip
from animator import Animator

output_dir = get_output_dir(__file__)
save_to_zip(output_dir)

with_animation = False
view_interval = 3000
view_duration = 150

max_steps = 300000

Rewards.escape = 1.0
Rewards.collision = -0.02

Robot.fov_size = 120
Robot.num_fov_pixels = 30
Robot.max_speed = 2
Robot.num_speeds = 3
Robot.turn_speed = 10

room = Room(
    width=14,
    height=22,
    door_width=6,
    max_episode_steps=None,
    history_save_interval=50000 # room steps
)
room.populate_with_robots(num_robots=8)

replay_buffer = PrioritizedReplayBuffer(name='rb1',size=100000,alpha=0.8)
dqn_policy = DQNPolicy(
    name='dqn1',
    state_shape=(Robot.num_fov_pixels, 2, 4),
    num_actions=3 * 3,
    widths=[128, 64, ],
    use_bn=False,
    replay_buffer_to=replay_buffer,
    beta0=0.4,
    beta_iters=300000, # opt steps
    last_n_steps=8,
    eps_start=0.9,
    eps_end=0.05,
    eps_decay=300000, # policy steps
    batch_size=32,
    gamma=0.99,
    optimization_interval=1,
    optimization_start=1,
    target_update_interval=1000, # opt steps
    lr=1e-3,
    optimizer='ADAM',
    checkpoint_save_interval=50000, # opt steps
)
# heuristic_policy = HeuristicPolicy(replay_buffer)

room.assign_policies([dqn_policy])

if with_animation:
    animator = Animator(room,pause_time=0.0001) # robots_to_debug=[room.robots[0]]
    animator.Update()

for step in count():
    if (max_steps is not None) and (step >= max_steps):
        break
    display = with_animation and ((step % view_interval) < view_duration)
    room.global_step(output_dir,inference=False)  # inference=display
    if display:
        animator.Update()
