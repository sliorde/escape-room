from itertools import count

from agent import Robot
from animator import Animator
from environment import Room
from policies import HeuristicPolicy, DQNPolicy
from utils import PrioritizedReplayBuffer

with_animation = True
view_interval = 3000
view_duration = 150

room = Room(
    width=14,
    height=22,
    door_width=6,
    max_episode_steps=None,
    history_save_interval=25000
)
room.populate_with_robots(num_robots=8)

replay_buffer = PrioritizedReplayBuffer(size=100000,alpha=0.8)
dqn_policy = DQNPolicy(
    name='dqn1',
    state_shape=(Robot.num_fov_pixels, 2, 4),
    num_actions=3 * 3,
    widths=[128, 64, ],
    use_bn=False,
    replay_buffer_to=replay_buffer,
    beta0=0.4,
    beta_iters=300000,
    last_n_steps=8,
    eps_start=0.9,
    eps_end=0.05,
    eps_decay=300000,
    batch_size=32,
    gamma=0.99,
    optimization_interval=1,
    optimization_start=1,
    target_update_interval=1000,
    lr=1e-3,
    optimizer='ADAM',
    checkpoint_save_interval=5000,
)
# heuristic_policy = HeuristicPolicy(replay_buffer)

room.assign_policies([dqn_policy])

if with_animation:
    animator = Animator(room) # robots_to_debug=[room.robots[0]]
    animator.Update()

for step in count():
    display = with_animation and ((step % view_interval) < view_duration)
    room.global_step(inference=False)  # inference=display
    if display:
        animator.Update()
