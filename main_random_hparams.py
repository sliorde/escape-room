from itertools import count
from math import log,exp,sqrt
import traceback

from agent import Robot
from environment import Room
from rewards import Rewards
from policies import HeuristicPolicy, DQNPolicy, SACPolicy
from utils import ReplayBuffer
from utils import get_output_dir, save_to_zip, save_params
from numpy.random import choice,uniform, randint

max_steps = 40004

class HParams():
    def __init__(self):
        self.dict = dict()
    def register(self, name, value):
        self.dict[name] = value
        return value
    def get(self):
        return self.dict
hparams = HParams()

while True:

    output_dir = get_output_dir(__file__)
    print(output_dir)
    save_to_zip(output_dir)

    Rewards.escape = 1.0
    Rewards.collision = hparams.register('collision_reward', uniform(-0.1, 0.0))

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
        history_save_interval=40000 # room steps
    )
    room.populate_with_robots(num_robots=12)

    replay_buffer = ReplayBuffer(size=100000)

    # dqn_policy = DQNPolicy(
    #     name='dqn1',
    #     state_shape=(Robot.num_fov_pixels, 2, 4),
    #     num_actions=3 * 3,
    #     widths=[192, 96, ],
    #     use_bn=False,
    #     affine_factor = 2/sqrt(room.width**2+room.height**2),
    #     affine_offset = -1,
    #     replay_buffer_base=replay_buffer,
    #     replay_buffer_alpha=hparams.register('replay_buffer_alpha', uniform(0.5, 1.0)),
    #     replay_buffer_beta0=hparams.register('replay_buffer_beta0', uniform(0.3, 0.7)),
    #     replay_buffer_beta_iters=hparams.register('replay_buffer_beta_iters', choice([100000, 200000, 300000, 400000, 500000])), # opt steps
    #     initial_max_priority=0.2,
    #     last_n_steps=len(room.robots),
    #     explore_eps_start=0.9,
    #     explore_eps_end=0.05,
    #     explore_eps_decay=hparams.register('explore_eps_decay',choice([100000,200000,300000,400000,500000])), # policy steps
    #     batch_size=hparams.register('batch_size',choice([32,64])),
    #     discount_gamma=hparams.register('discount_gamma',choice([0.91,0.94,0.97,0.99])),
    #     optimization_interval=1,
    #     optimization_start=1,
    #     target_update_interval=hparams.register('target_update_interval',choice([1000,5000,10000])), # opt steps
    #     lr=hparams.register('lr',choice([1e-4,3e-4,1e-3,3e-3,1e-2,3e-2])),
    #     optimizer='ADAM',
    #     checkpoint_save_interval=50000, # opt steps
    # )
    sac_policy = SACPolicy(
        name='sac1',
        state_shape=(Robot.num_fov_pixels, 2, 4),
        num_actions=3 * 3,
        widths=[192, 96, ],
        use_bn=False,
        affine_factor = 2/sqrt(room.width**2+room.height**2),
        affine_offset = -1,
        replay_buffer_base=replay_buffer,
        replay_buffer_alpha=hparams.register('replay_buffer_alpha', uniform(0.5, 1.0)),
        replay_buffer_beta0=hparams.register('replay_buffer_beta0', uniform(0.3, 0.7)),
        replay_buffer_beta_iters=hparams.register('replay_buffer_beta_iters', choice([100000, 200000, 300000, 400000, 500000])), # opt steps
        initial_max_priority=0.2,
        target_entropy=hparams.register('target_entropy',uniform(0.3,1.3)),
        target_ema_rate=hparams.register('target_ema_rate',exp(uniform(log(3e-4),log(3e-2)))),
        batch_size=hparams.register('batch_size',choice([32,64])),
        discount_gamma=hparams.register('discount_gamma',choice([0.91,0.94,0.97,0.99])),
        optimization_interval=1,
        optimization_start=500,
        lr=hparams.register('lr',choice([1e-4,3e-4,1e-3,3e-3,1e-2,3e-2])),
        lr_temperature=hparams.register('lr_temperature',choice([1e-4,3e-4,1e-3,3e-3,1e-2,3e-2])),
        optimizer='ADAM',
        checkpoint_save_interval=40000,
        last_n_steps=hparams.register('last_n_steps',choice([0,len(room.robots)])),
    )

    policy = sac_policy

    # heuristic_policy = HeuristicPolicy(replay_buffer)

    save_params(output_dir, width=room.width, height=room.height, door_width=room.door_width, radius=Robot.radius, gamma=policy.discount_gamma, escape_reward=Rewards.escape, **hparams.get())

    room.assign_policies([policy])

    for step in count():
        if (max_steps is not None) and (step >= max_steps):
            break
        room.global_step(output_dir,inference=False)  # inference=display