# -*- coding: utf-8 -*-
import torch
import numpy as np 
import gym
import yaml


def read_yaml_file(file_path):
    with open(file_path, 'r') as file:
        conf = yaml.safe_load(file)
    return conf


class TimeDelayEnv(gym.Wrapper):
    def __init__(self, env, delay):
        """A way to add control delay. you can also do it inside OOPAO or OOPAO_env. ALL GYM stuff is OPTIONAL
        I quicly coded this. Debug this if you need it"""

        super(TimeDelayEnv, self).__init__(env)
        self._env = env       
        self.d = delay
        self.action_buffer = [np.zeros(env.action_space.shape)] * delay
        

    def reset(self, new_atmos = True):
        obs = self._env.reset(new_atmos)
        self.action_buffer = [np.zeros(self._env.action_space.shape)] * self.d
        return obs

    def reset_soft(self):
        obs = self._env.reset_soft()
        self.action_buffer = [np.zeros(self._env.action_space.shape)] * self.d
        return obs    

    def step(self, action):
        self.action_buffer.append(action)
        obs, reward, done, info = self._env.step(self.action_buffer[0])

        del self.action_buffer[0]

        return obs, reward, done, info


class EfficientExperienceReplay():

    def __init__(self, state_shape, action_shape, max_size=100000):
        self.max_size = max_size

        self.states = torch.empty(max_size, *state_shape)
        self.next_states = torch.empty(max_size, *state_shape)
        self.actions = torch.empty(max_size, *action_shape)
        self.rewards = torch.empty(max_size, 1)

        self.len = 0

    def add(self, replay):
        cur_len = self.len
        new_len = self.len + len(replay)
        
        if isinstance(replay, EfficientExperienceReplay):
            replay = ReplaySample(replay.states[:len(replay)], replay.actions[:len(replay)], replay.rewards[:len(replay)], replay.next_states[:len(replay)])
        
        self.states[cur_len:new_len] = replay.state()
        self.next_states[cur_len:new_len] = replay.next_state()
        self.actions[cur_len:new_len] = replay.action()
        self.rewards[cur_len:new_len] = replay.reward()

        self.len = new_len

    def __add__(self, replay):
        self.add(replay)
        return self
    
    def append(self, obs, action, reward, next_obs, done):

        if isinstance(obs, np.ndarray):
            raise 'should be torch'
            #state, prev_action, next_state, action = torch.from_numpy(state).float(), torch.from_numpy(prev_action).float(), torch.from_numpy(next_state).float(), torch.from_numpy(action).float()

        self.states[self.len] = obs
        self.next_states[self.len] = next_obs
        self.actions[self.len] = action
        self.rewards[self.len] = reward

        self.len += 1

    def sample_contiguous(self, horizon, max_ts, batch_size=32):    
        inds = torch.randint(0, max_ts - (horizon + 1), size=(batch_size, ))
        inds += torch.randint(0, len(self) // max_ts, size=(batch_size, )) * max_ts
        
        indices = torch.cat([torch.arange(ind, ind + horizon + 1) for ind in inds])
        # TODO check correct
        #indices = torch.from_numpy(vrange(inds.numpy(), np.ones_like(inds) * horizon + 1))
        
        return ReplaySample(self.states[indices], self.actions[indices], self.rewards[indices], self.next_states[indices])

    def reward(self):
        return self.rewards[:self.len]
    
    def next_state(self):
        return self.next_states[:self.len]
    
    def state(self):
        return self.states[:self.len]

    def action(self):
        return self.actions[:self.len]

    def __len__(self):
        return self.len

    def sample(self, size=512):
        inds = torch.randperm(self.len)[:size]
        return ReplaySample(self.states[inds], self.actions[inds], self.rewards[inds], self.next_states[inds])  

    def clear(self):
        self.len = 0

class ReplaySample():
    def __init__(self, states, actions, rewards, next_states):
        self.states = states
        self.next_states = next_states
        self.actions = actions
        self.rewards = rewards

    def state(self):
        return self.states
    
    def prev_action(self):
        return self.prev_actions

    def next_state(self):
        return self.next_states 

    def action(self):
        return self.actions

    def reward(self):
        return self.rewards

    def __len__(self):
        return len(self.states)

    def to(self, device):
        self.states = self.states.to(device)
        self.next_states = self.next_states.to(device)
        self.actions = self.actions.to(device)
        self.rewards = self.rewards.to(device)
        return self

import contextlib
import os

@contextlib.contextmanager
def stdchannel_redirected(stdchannel, dest_filename):
    """
    A context manager to temporarily redirect stdout or stderr

    e.g.:


    with stdchannel_redirected(sys.stderr, os.devnull):
        if compiler.has_function('clock_gettime', libraries=['rt']):
            libraries.append('rt')
    """

    try:
        oldstdchannel = os.dup(stdchannel.fileno())
        dest_file = open(dest_filename, 'w')
        os.dup2(dest_file.fileno(), stdchannel.fileno())

        yield
    finally:
        if oldstdchannel is not None:
            os.dup2(oldstdchannel, stdchannel.fileno())
        if dest_file is not None:
            dest_file.close()

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp



class TorchWrapper(gym.Wrapper):
    """
    HERE I only convert to from torch to numpy and back. Image to vec is handeled in the enviroment.
    """
    def __init__(self, env):
        super().__init__(env)
        self._env = env
    
    def step(self, i,action):
        action_to_numpy = action.numpy()
        next_obs, reward,strehl, done, info = self._env.step(i,action_to_numpy) 

        return torch.tensor(next_obs, dtype=torch.float32), reward,strehl, done, [(k, torch.tensor(v, dtype=torch.float32)) for k, v in info.items()]
    
    def reset(self):
        obs = self._env.reset()
        return torch.tensor(obs, dtype=torch.float32)

    def reset_soft(self):
        obs= self._env.reset_soft()
        return torch.tensor(obs, dtype=torch.float32)



