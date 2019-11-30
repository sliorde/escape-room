import random
from collections import namedtuple

import torch

from utils import encode_action

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

        self.prev_state = {}
        self.prev_action = {}
        self.prev_reward = {}

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

    def add_to_replay_buffer(self, state, action, reward, robot, ):
        if state is not None:
            state = torch.tensor(state).flatten()
        prev_state = self.prev_state.get(robot, None)
        if prev_state is not None:
            self.push(prev_state, self.prev_action[robot], state, self.prev_reward[robot])
        self.prev_state[robot] = state
        self.prev_action[robot] = torch.tensor(encode_action(*action))
        self.prev_reward[robot] = torch.tensor(reward)
