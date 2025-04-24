#%%
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import random
import time
from dataclasses import dataclass

import matplotlib.pyplot as plt
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from OOPAOEnv.learnIMEnv import OOPAO
from learnIM import Actor

env = OOPAO()

def make_env():
    def thunk():
        env = OOPAO()
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    return thunk

envs = gym.vector.SyncVectorEnv([make_env()])

actor = Actor(envs)

#actor.load_state_dict(torch.load("./RL/models/best_model.pth"))
# %%
# Residuals with turbulence

# Reset the environment
obs, info = env.reset(seed=0)


residuals_turbulence = []
# Start the loop
for i in range(env.nLoops):
    # Take a step in the environment
    obs, reward, terminated, truncated, info = env.step(np.zeros_like(obs))

    residuals_turbulence.append(reward)

# Residuals with integrator

# Reset the environment
obs, info = env.reset(seed=0)

residuals_integrator = []
# Start the loop
for i in range(env.nLoops):
    #Integrator action
    action = -1 * obs
    # Take a step in the environment
    obs, reward, terminated, truncated, info = env.step(action)

    residuals_integrator.append(reward)


# Residuals with actor

obs, info = env.reset(seed=0)
residuals_actor = []

# Start the loop
for i in range(env.nLoops):
    # Take a step in the environment
    actions, _, _ = actor.get_action(torch.Tensor(obs))

    obs, reward, terminated, truncated, info = env.step(action)

    residuals_actor.append(reward)


