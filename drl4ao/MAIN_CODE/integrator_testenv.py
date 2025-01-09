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
from VPG.callbacks import CustomEvalCallback
from stable_baselines3 import PPO, SAC

#%%
train_env = OOPAO()
eval_env = OOPAO()

#%%

state_dim = train_env.observation_space.shape
action_dim = train_env.action_space.shape

policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=512),
)

custom_eval_callback = CustomEvalCallback(eval_env, eval_freq=1e4)

model = PPO("CnnPolicy", train_env, 
            learning_rate=1e-4, clip_range=0.3,
            policy_kwargs=policy_kwargs, verbose=1,
            tensorboard_log="./ppo_ao_tensorboard/")

model.learn(total_timesteps=int(2e5), progress_bar=True, callback=custom_eval_callback)

model.save("PPO_oopao")
# %%
