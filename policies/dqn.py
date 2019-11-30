import random

import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from policies import Policy
from utils import decode_action, Transition


class DQN(nn.Module):

    def __init__(self, state_numel, num_actions, widths):
        nn.Module.__init__(self)
        widths = [state_numel] + widths + [num_actions]
        layers = []
        layers.append(nn.BatchNorm1d(state_numel))
        for i, (w1, w2) in enumerate(zip(widths[:(-1)], widths[1:])):
            with_bn = i < (len(widths) - 2)
            layers.append(nn.Linear(w1, w2, bias=not with_bn))
            if with_bn:
                layers.append(nn.BatchNorm1d(w2))
                layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)

    def forward(self, state):
        return self.layers(state)


class DQNPolicy(nn.Module, Policy):
    def __init__(self, state_shape, num_actions, widths, eps_start, eps_end, eps_decay, batch_size, gamma,
                 optimization_interval, target_update_interval, lr, optimizer,replay_buffer_to, replay_buffer_from=None):
        nn.Module.__init__(self)
        Policy.__init__(self, replay_buffer_to, replay_buffer_from)

        self.policy_net = DQN(np.prod(state_shape), num_actions, widths)
        self.target_net = DQN(np.prod(state_shape), num_actions, widths)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.policy_net.eval()
        self.target_net.eval()

        self.steps_done = 0
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay

        self.num_actions = num_actions
        self.batch_size = batch_size
        self.gamma = gamma
        self.optimization_interval = optimization_interval
        self.target_update_interval = target_update_interval

        if optimizer.casefold()=='ADAM'.casefold():
            self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr)
        else:
            self.optimizer = torch.optim.SGD(self.policy_net.parameters(), lr)

        self.prev_state = {}
        self.prev_action = {}
        self.prev_reward = {}

    def choose_action_inference(self, state):
        # device = self.get_device()
        # state = torch.from_numpy(state).to(device).flatten()
        state = torch.from_numpy(state)
        with torch.no_grad():
            a = self.policy_net(state.flatten()[None, :]).argmax(1)
            self.steps_done += 1
            return decode_action(int(a))

    def choose_action_training(self, state):
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
                        math.exp(-1. * self.steps_done / self.eps_decay)
        if sample > eps_threshold:
            return self.choose_action_inference(state)
        else:
            self.steps_done += 1
            return decode_action(random.randint(0, self.num_actions - 1))

    def optimization_step(self):
        if (self.steps_done > 0) and (self.steps_done % self.optimization_interval == 0):
            device = self.get_device()
            if len(self.replay_buffer_from) < self.batch_size:
                return
            transitions = self.replay_buffer_from.sample(self.batch_size)
            batch = Transition(*zip(*transitions))
            non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device,
                                          dtype=torch.uint8)
            non_final_next_states = torch.stack([s for s in batch.next_state if s is not None])
            state_batch = torch.stack(batch.state)
            action_batch = torch.stack(batch.action)
            reward_batch = torch.stack(batch.reward)

            self.policy_net.train()
            state_action_values = self.policy_net(state_batch).gather(1, action_batch[:, None])
            next_state_values = torch.zeros(self.batch_size, device=device)
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
            expected_state_action_values = (next_state_values * self.gamma) + reward_batch

            # Compute Huber loss
            loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

            # Optimize the model
            self.optimizer.zero_grad()
            loss.backward()
            # for param in self.policy_net.parameters():
            #     param.grad.data.clamp_(-1, 1)
            self.optimizer.step()
            self.policy_net.eval()

            if (self.steps_done > 0) and (self.steps_done % self.target_update_interval == 0):
                self.target_net.load_state_dict(self.policy_net.state_dict())

    def get_device(self):
        return next(self.policy_net.parameters()).device
