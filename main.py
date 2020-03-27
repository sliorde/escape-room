from itertools import count
from math import log,sqrt

from agent import Robot
from environment import Room
from rewards import Rewards
from policies import HeuristicPolicy, DQNPolicy, SACPolicy
from utils import ReplayBuffer
from utils import get_output_dir, save_to_zip, save_params
from animator import Animator

output_dir = get_output_dir(__file__)
save_to_zip(output_dir)

with_animation = False
view_interval = 5000
view_duration = 150

max_steps = 500000

Rewards.escape = 1.0
Rewards.collision = -0.05208921768082558

Robot.fov_size = 170
Robot.num_fov_pixels = 50
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
room.populate_with_robots(num_robots=12)

replay_buffer = ReplayBuffer(size=100000)

# dqn_policy = DQNPolicy(
#     name='dqn1',
#     state_shape=(Robot.num_fov_pixels, 2, 4),
#     num_actions=3 * 3,
#     widths=[256, 128, ],
#     use_bn=False,
#     affine_factor = 2/sqrt(room.width**2+room.height**2),
#     affine_offset = -1,
#     replay_buffer_base=replay_buffer,
#     replay_buffer_alpha=0.7222068575401144,
#     replay_buffer_beta0=0.6083732563905723,
#     replay_buffer_beta_iters=200000, # opt steps
#     initial_max_priority = 0.2,
#     last_n_steps=len(room.robots),
#     explore_eps_start=0.9,
#     explore_eps_end=0.05,
#     explore_eps_decay=400000, # policy steps
#     batch_size=64,
#     discount_gamma=0.99,
#     optimization_interval=1,
#     optimization_start=1,
#     target_update_interval=1000, # opt steps
#     lr=1e-4,
#     optimizer='ADAM',
#     checkpoint_save_interval=50000, # opt steps
# )
sac_policy = SACPolicy(
    name='sac1',
    state_shape=(Robot.num_fov_pixels, 2, 4),
    num_actions=3 * 3,
    widths=[128, 64, ],
    use_bn=False,
    affine_factor=2 / sqrt(room.width ** 2 + room.height ** 2),
    affine_offset=-1,
    replay_buffer_base=replay_buffer,
    replay_buffer_alpha=0.7492659346586595,
    replay_buffer_beta0=0.6318652575196004,
    replay_buffer_beta_iters=100000,  # opt steps
    initial_max_priority=0.2,
    target_entropy=1.2778316560993492,
    target_ema_rate=0.0003721987634250899,
    batch_size=64,
    discount_gamma=0.97,
    optimization_interval=1,
    optimization_start=1,
    lr=0.01,
    lr_temperature=0.01,
    optimizer='ADAM',
    checkpoint_save_interval=40000,
    last_n_steps=0,
)

policy = sac_policy

# heuristic_policy = HeuristicPolicy(replay_buffer)

save_params(output_dir, width=room.width, height=room.height, door_width=room.door_width, radius=Robot.radius, gamma=policy.discount_gamma, escape_reward=Rewards.escape)

room.assign_policies([policy])

if with_animation:
    animator = Animator(room,pause_time=0.0001)#,robots_to_debug=[room.robots[0]])
    animator.Update()

for step in count():
    if (max_steps is not None) and (step >= max_steps):
        break
    display = with_animation and ((step % view_interval) < view_duration)
    room.global_step(output_dir,inference=False)  # inference=display
    if display:
        animator.Update()
