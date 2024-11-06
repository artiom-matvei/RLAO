#%%
import torch
import numpy as np
import os, sys
import matplotlib.pyplot as plt
from score_models import ScoreModel, NCSNpp

from data_loading.dataset_tools import DiffusionDataset, uncondDataset



sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from PO4AO.mbrl import get_env
from ML_stuff.dataset_tools import read_yaml_file
from types import SimpleNamespace
try:
    args = SimpleNamespace(**read_yaml_file('./Conf/papyrus_config.yaml'))
except:
    args = SimpleNamespace(**read_yaml_file('../Conf/papyrus_config.yaml'))


env = get_env(args)
# %%

# Print the size in mb of the dataset file
lr = np.load('/Users/parkerlevesque/School/Research/AO/RLAO/drl4ao/MAIN_CODE/diffusion/images/lr.npy')
hr = np.load('/Users/parkerlevesque/School/Research/AO/RLAO/drl4ao/MAIN_CODE/diffusion/images/hr.npy')

data = uncondDataset('/Users/parkerlevesque/School/Research/AO/RLAO/drl4ao/MAIN_CODE/diffusion/images/lr.npy', (48,48), device='cpu', size=100)

model = ScoreModel(checkpoints_directory='/Users/parkerlevesque/School/Research/AO/RLAO/drl4ao/MAIN_CODE/diffusion/cp_unconditional/')

B = 1
channels = 4

start_from_y = False
# %%
num_steps = 1000
dt = -1/num_steps #reverse time

s_min = model.sde.sigma_min
s_max = model.sde.sigma_max

eta = 0.1

y = data.__getitem__(0).unsqueeze(0)

x_t = torch.normal(0, s_max, (B, channels, 24, 24))

t_start = 1 if not start_from_y else np.log(np.std(lr[0][0] - hr[0][0])) / np.log(s_max / s_min)

for t in np.linspace(t_start, 0, num_steps):
    t = torch.tensor(t) * torch.ones(B)
    z = torch.randn_like(x_t)
    dw = abs(dt)**(1/2) * z
    g = model.sde.diffusion(t, x_t)

    score_likelihood = - (y - x_t) / (model.sde.sigma(t) ** 2 + eta**2)
    score_prior = model.score(t, x_t)

    dx = - g**2 * ( score_likelihood + score_prior) * dt + g * dw

    x_t += dx


# %%
