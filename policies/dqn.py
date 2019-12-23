import random
import math
import pickle
from os import path
import zlib

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from policies import Policy
from utils import decode_action, FixedAffine, PrioritizedReplayBuffer

class QNetwork(nn.Module):

    def __init__(self, state_numel, num_actions, widths, use_bn, affine_factor,affine_offset):
        nn.Module.__init__(self)
        widths = [state_numel] + widths + [num_actions]
        layers = []
        layers.append(FixedAffine(affine_factor,affine_offset))
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


class DQNPolicy(Policy):
    def __init__(self, state_shape, num_actions, widths, use_bn, affine_factor,affine_offset, replay_buffer_base, replay_buffer_alpha, replay_buffer_beta0, replay_buffer_beta_iters,initial_max_priority,explore_eps_start, explore_eps_end, explore_eps_decay, batch_size, discount_gamma, optimization_interval, optimization_start, target_update_interval, lr, optimizer, checkpoint_save_interval, last_n_steps, name=None):
        replay_buffer = PrioritizedReplayBuffer(replay_buffer_base,replay_buffer_alpha,initial_max_priority,name)

        Policy.__init__(self, replay_buffer)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.policy_net = QNetwork(np.prod(state_shape), num_actions, widths, use_bn,affine_factor,affine_offset).to(device)
        self.target_net = QNetwork(np.prod(state_shape), num_actions, widths, use_bn,affine_factor,affine_offset).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.policy_net.eval()
        self.target_net.eval()

        self.env_steps_done = 0
        self.opt_steps_done = 0
        self.explore_eps_start = explore_eps_start
        self.explore_eps_end = explore_eps_end
        self.explore_eps_decay = explore_eps_decay

        self.num_actions = num_actions
        self.batch_size = batch_size
        self.discount_gamma = discount_gamma
        self.optimization_interval = optimization_interval
        self.optimization_start = optimization_start
        self.target_update_interval = target_update_interval
        self.replay_buffer_beta0 = replay_buffer_beta0
        self.replay_buffer_beta_iters = replay_buffer_beta_iters
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
        eps_threshold = self.explore_eps_end + (self.explore_eps_start - self.explore_eps_end) * \
                        math.exp(-1. * self.env_steps_done / self.explore_eps_decay)
        if sample > eps_threshold:
            return self.choose_action_inference(state)
        else:
            self.env_steps_done += 1
            return decode_action(int(random.randint(0, self.num_actions - 1)))

    def save_checkpoint(self,output_dir):
        cp_to_save = {
            'policy_net':self.policy_net.state_dict(),
            'target_net':self.target_net.state_dict(),
            'optimizer':self.optimizer.state_dict(),
            'env_steps_done':self.env_steps_done,
            'opt_steps_done':self.opt_steps_done,
            'explore_eps_start':self.explore_eps_start,
            'explore_eps_end':self.explore_eps_end,
            'explore_eps_decay':self.explore_eps_decay,
            'num_actions':self.num_actions,
            'batch_size':self.batch_size,
            'gamma':self.discount_gamma,
            'optimization_interval':self.optimization_interval,
            'optimization_start':self.optimization_start,
            'target_update_interval':self.target_update_interval,
            'replay_buffer_beta0':self.replay_buffer_beta0,
            'replay_buffer_beta_iters':self.replay_buffer_beta_iters,
            'last_n_steps':self.last_n_steps,
            'checkpoint_save_interval': self.checkpoint_save_interval
        }
        torch.save(cp_to_save,path.join(output_dir,'checkpoint{:s}_{:d}.ckpt'.format('_'+self.name if self.name is not None else '',self.opt_steps_done)))

        rb_to_save = {
            'replay_buffer': pickle.dumps(self.replay_buffer),
        }
        with open(path.join(output_dir,'replay_buffer_{:s}'.format(self.replay_buffer.name)),'wb') as f:
            f.write(zlib.compress(pickle.dumps(rb_to_save)))

    def optimization_step(self,output_dir):
        if (self.env_steps_done >= self.optimization_start) and ((self.env_steps_done-self.optimization_start) % self.optimization_interval == 0):
            device = self.get_device()
            if len(self.replay_buffer) < self.batch_size:
                return
            beta = np.clip(self.replay_buffer_beta0 + (1.0 - self.replay_buffer_beta0) * (self.opt_steps_done / self.replay_buffer_beta_iters), self.replay_buffer_beta0, 1.0)
            states, actions, rewards, next_states, is_final_states, weights, batch_inds  = self.replay_buffer.sample(self.batch_size, beta, self.last_n_steps)
            states = torch.from_numpy(states).to(device)
            actions = torch.from_numpy(actions).to(device)
            rewards = torch.from_numpy(rewards).to(device)
            next_states = torch.from_numpy(next_states).to(device)
            is_non_final_states = ~torch.from_numpy(is_final_states).bool().to(device)
            weights = torch.from_numpy(weights).to(device)

            self.policy_net.train()
            state_action_values = self.policy_net(states.reshape(states.shape[0],-1)).gather(1, actions[:, None]).squeeze(1)
            next_state_values = torch.zeros(self.batch_size, device=device)
            next_state_values[is_non_final_states] = self.target_net(next_states.reshape(next_states.shape[0],-1)[is_non_final_states]).max(1)[0].detach()
            expected_state_action_values = rewards + (next_state_values * self.discount_gamma)
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
            self.replay_buffer.update_priorities(batch_inds,new_priorities)

            self.opt_steps_done += 1

            if (self.checkpoint_save_interval is not None) and (self.opt_steps_done % self.checkpoint_save_interval == 0):
                self.save_checkpoint(output_dir)

    def get_device(self):
        return next(self.policy_net.parameters()).device
