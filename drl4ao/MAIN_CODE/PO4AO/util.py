# -*- coding: utf-8 -*-
import torch
import numpy as np
from torch import nn
import gym
from gym import spaces

class Flatten(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        bs = x.shape[0]
        return x.view(bs, -1)

class BaseNet(nn.Module):

    def __init__(self, n_prev_actions=1):
        super().__init__()

        n_input_channels = 2
        k_size = 3

        self.net = nn.Sequential(
            nn.Conv2d(n_input_channels, 64, k_size, padding=(k_size // 2)),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, k_size, padding=(k_size // 2)),
            nn.LeakyReLU(),
            nn.Conv2d(64, 32, k_size, padding=(k_size // 2)),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.net(x)


class Dynamics(nn.Module):
    def __init__(self, xvalid, yvalid):
        super().__init__()

        n_input_channels = 3
        k_size = 3
        self.xvalid = xvalid
        self.yvalid = yvalid

        self.net = nn.Sequential(
            nn.Conv2d(n_input_channels, 64, k_size, padding=(k_size // 2)),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, k_size, padding=(k_size // 2)),
            nn.LeakyReLU(),
            nn.Conv2d(64, 32, k_size, padding=(k_size // 2)),
            nn.LeakyReLU()
        )

        self.output = nn.Sequential(
            nn.Conv2d(32, 1, 3, padding=1),
            #nn.Tanh()
        )

    # TODO preprocess mask
    def forward(self, state, action):
        if action.ndim == 4:
            action_img = action
        else:
            action_img = torch.zeros(len(action), 1, 17, 17)
            action_img[..., self.xvalid, self.yvalid] = action.view(action.shape[0], 1, 220)
        feats = self.net(torch.cat([state, action_img], dim=1))
        mean = self.output(feats)
        mask = torch.zeros_like(mean, dtype=torch.bool)
        mask[:, :, self.xvalid, self.yvalid] = 1
        return mean * mask

class Policy(nn.Module):
    def __init__(self, env, xvalid, yvalid, stochastic=True):
        super().__init__()

        self.base_net = BaseNet()
        self.policy_actuators = nn.Sequential(
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Tanh()
        )
        self.xvalid = torch.from_numpy(xvalid)
        self.yvalid = torch.from_numpy(yvalid)

    def forward(self, state, sigma=0.0):
        feats = self.base_net(state)
        actuators = self.policy_actuators(feats)

        actuators = actuators[:, :, self.xvalid, self.yvalid].squeeze(1)

        return (actuators + sigma * torch.randn_like(actuators))#.clamp(-1, 1)

class TorchWrapperNoDelay(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self._env = env
        self.action_space = spaces.Box(-5, 5, shape=(self._env.n_actu, self._env.n_actu, ), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(1, self._env.n_actu, self._env.n_actu), dtype=np.float32)
        #self.normalizer = np.load('int_std.npy').mean()* 2.5/8
        self.normalizer = 1
        self.F = np.identity(self._env.action_space.sample().shape[0])

        self.last_action = self._env.img_to_vec(np.zeros(self.action_space.shape))

    def step(self, action):
        action_to_numpy = action.numpy()
        action_vec = self._env.img_to_vec(action_to_numpy)
        #action_vec = np.clip(self._env.F @ (action_vec * self.normalizer + self.last_action),
        action_vec = np.clip((action_vec * self.normalizer + self.last_action),-170,170)

        #print(action_vec.shape)

        next_obs, reward, done, info = self._env.step(action_vec)
        next_obs = self.process_obs(next_obs)

        self.last_action = action_vec

        #self.last_action = -1 * np.clip(action_vec * self.normalizer + self.last_action,-170,170)

        return torch.tensor(next_obs, dtype=torch.float32), reward, done, [(k, torch.tensor(v, dtype=torch.float32)) for k, v in info.items()]

    def reset(self, new_atmos = True):
        obs = self._env.reset()
        obs = self.process_obs(obs)

        return torch.tensor(obs, dtype=torch.float32)

    def reset_soft(self):
        obs= self._env.reset(soft=True)
        obs = self.process_obs(obs)
        return torch.tensor(obs, dtype=torch.float32)

    def process_obs(self,obs):
        obs = self._env.vec_to_img(-1 * self._env.get_S2V() @ obs)
        obs = np.expand_dims(obs, 0) #/ self.normalizer

        return obs

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp
