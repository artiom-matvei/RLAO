"""
OOPAO module for the integrator
"""

#%%
import os,sys
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
# from parser_Configurations import Config, ConfigAction
# from OOPAOEnv.OOPAOEnvRazor import OOPAO
from ML_stuff.dataset_tools import read_yaml_file #TorchWrapper, 
from ML_stuff.models import Unet_big
from Plots.gifTools import create_gif

import time
import numpy as np

from types import SimpleNamespace
import matplotlib.pyplot as plt
from OOPAOEnv.OOPAOEnv_VPG import OOPAO
from VPG.policy_networks import CustomCNN
from stable_baselines3 import PPO, SAC

#%%
env = OOPAO()

state_dim = env.observation_space.shape
action_dim = env.action_space.shape

policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=128),
)


model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
model.learn(total_timesteps=int(2e4), progress_bar=True)


# %%
