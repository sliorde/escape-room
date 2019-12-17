import random
import math
import pickle
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from policies import Policy
from utils import decode_action, FixedAffine

class DQN(nn.Module):

    def __init__(self, state_numel, num_actions, widths, use_bn):
        nn.Module.__init__(self)
        widths = [state_numel] + widths + [num_actions]
        layers = []
        layers.append(FixedAffine(0.1,-1))
        # if use_bn:
        #     layers.append(nn.BatchNorm1d(state_numel))
        for i, (w1, w2) in enumerate(zip(widths[:(-1)], widths[1:])):
            last = (i==(len(widths) - 2))
            with_bn = use_bn and not last
            layers.append(nn.Linear(w1, w2, bias=not with_bn))
            if with_bn:
                layers.append(nn.BatchNorm1d(w2))
            if not last:
                layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)

    def forward(self, state):
        return self.layers(state)


class DQNPolicy(nn.Module, Policy):
    def __init__(self, state_shape, num_actions, widths, use_bn, eps_start, eps_end, eps_decay, batch_size, gamma, optimization_interval, optimization_start, target_update_interval, lr, optimizer, checkpoint_save_interval, replay_buffer_to, beta0, beta_iters, last_n_steps, replay_buffer_from=None,name=None):
        nn.Module.__init__(self)
        Policy.__init__(self, replay_buffer_to, replay_buffer_from)

        self.policy_net = DQN(np.prod(state_shape), num_actions, widths, use_bn)
        self.target_net = DQN(np.prod(state_shape), num_actions, widths, use_bn)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.policy_net.eval()
        self.target_net.eval()

        self.env_steps_done = 0
        self.opt_steps_done = 0
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay

        self.num_actions = num_actions
        self.batch_size = batch_size
        self.gamma = gamma
        self.optimization_interval = optimization_interval
        self.optimization_start = optimization_start
        self.target_update_interval = target_update_interval
        self.beta0 = beta0
        self.beta_iters = beta_iters
        self.last_n_steps = last_n_steps

        if optimizer.casefold()=='ADAM'.casefold():
            self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr)
        else:
            self.optimizer = torch.optim.SGD(self.policy_net.parameters(), lr)

        self.checkpoint_save_interval = checkpoint_save_interval
        self.name = name

    def choose_action_inference(self, state):
        device = self.get_device()
        # state = torch.from_numpy(state).to(device).flatten()
        state = torch.from_numpy(state).to(device)
        with torch.no_grad():
            a = self.policy_net(state.flatten()[None, :]).argmax(1)
            self.env_steps_done += 1
            return decode_action(int(a))

    def choose_action_training(self, state):
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
                        math.exp(-1. * self.env_steps_done / self.eps_decay)
        if sample > eps_threshold:
            return self.choose_action_inference(state)
        else:
            self.env_steps_done += 1
            return decode_action(int(random.randint(0, self.num_actions - 1)))

    def save_checkpoint(self):
        to_save = {
            'policy_net':self.policy_net.state_dict(),
            'target_net':self.target_net.state_dict(),
            'optimizer':self.optimizer.state_dict(),
            'env_steps_done':self.env_steps_done,
            'opt_steps_done':self.opt_steps_done,
            'eps_start':self.eps_start,
            'eps_end':self.eps_end,
            'eps_decay':self.eps_decay,
            'num_actions':self.num_actions,
            'batch_size':self.batch_size,
            'gamma':self.gamma,
            'optimization_interval':self.optimization_interval,
            'optimization_start':self.optimization_start,
            'target_update_interval':self.target_update_interval,
            'beta0':self.beta0,
            'beta_iters':self.beta_iters,
            'last_n_steps':self.last_n_steps,
            'checkpoint_save_interval': self.checkpoint_save_interval,
            'replay_buffer_to':pickle.dumps(self.replay_buffer_to),
            'replay_buffer_from': None if self.replay_buffer_from is self.replay_buffer_to else pickle.dumps(self.replay_buffer_from),
        }
        os.makedirs('checkpoints',exist_ok=True)
        torch.save(to_save,'checkpoints/checkpoint{:s}.ckpt'.format('_'+self.name if self.name is not None else ''))

    def optimization_step(self):
        if (self.env_steps_done >= self.optimization_start) and ((self.env_steps_done-self.optimization_start) % self.optimization_interval == 0):
            device = self.get_device()
            if len(self.replay_buffer_from) < self.batch_size:
                return
            beta = np.clip(self.beta0 + (1.0 - self.beta0) * (self.opt_steps_done / self.beta_iters), self.beta0, 1.0)
            states, actions, rewards, next_states, is_final_states, weights, batch_inds  = self.replay_buffer_from.sample(self.batch_size,beta,self.last_n_steps)
            states = torch.from_numpy(states).to(device)
            actions = torch.from_numpy(actions).to(device)
            rewards = torch.from_numpy(rewards).to(device)
            next_states = torch.from_numpy(next_states).to(device)
            is_non_final_states = ~torch.from_numpy(is_final_states).to(device)
            weights = torch.from_numpy(weights).to(device)

            self.policy_net.train()
            state_action_values = self.policy_net(states.reshape(states.shape[0],-1)).gather(1, actions[:, None]).squeeze(1)
            next_state_values = torch.zeros(self.batch_size, device=device)
            next_state_values[is_non_final_states] = self.target_net(next_states.reshape(next_states.shape[0],-1)[is_non_final_states]).max(1)[0].detach()
            expected_state_action_values = rewards + (next_state_values * self.gamma)
            td_errors = (state_action_values - expected_state_action_values).detach()

            # Compute Huber loss
            losses = F.smooth_l1_loss(state_action_values, expected_state_action_values,reduction='none')
            loss = torch.mean(weights*losses)

            # Optimize the model
            self.optimizer.zero_grad()
            loss.backward()
            # for param in self.policy_net.parameters():
            #     param.grad.data.clamp_(-1, 1)
            self.optimizer.step()
            self.policy_net.eval()

            if (self.opt_steps_done > 0) and (self.opt_steps_done % self.target_update_interval == 0):
                self.target_net.load_state_dict(self.policy_net.state_dict())

            new_priorities = np.abs(td_errors.cpu().numpy())+1e-6
            self.replay_buffer_from.update_priorities(batch_inds,new_priorities)

            self.opt_steps_done += 1

            if (self.checkpoint_save_interval is not None) and (self.opt_steps_done % self.checkpoint_save_interval == 0):
                self.save_checkpoint()

    def get_device(self):
        return next(self.policy_net.parameters()).device
