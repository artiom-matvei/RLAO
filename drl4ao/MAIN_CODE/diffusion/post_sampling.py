#%%
import torch
import numpy as np
import os, sys
import matplotlib.pyplot as plt
from score_models import ScoreModel, NCSNpp

# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

script_dir = os.path.dirname(os.path.abspath(__file__))
# Print the size in mb of the dataset file
lr = np.load(f'{script_dir}/images/lr.npy')
hr = np.load(f'{script_dir}/images/hr.npy')

model = ScoreModel(checkpoints_directory=f'{script_dir}/cp_unconditional/')

B = 100
channels = 4

start_from_y = False
# %%
num_steps = 1000
dt = -1/num_steps #reverse time

s_min = model.sde.sigma_min
s_max = model.sde.sigma_max

eta = 0.1

y = torch.from_numpy(lr).to(device)

x_t = torch.normal(0, s_max, (B, channels, 24, 24)).to(device)

t_start = 1 # if not start_from_y else np.log(np.std(lr[0][0] - hr[0][0])) / np.log(s_max / s_min)

for t in np.linspace(t_start, 0, num_steps):
    print(t)
    t = torch.tensor(t).to(device) * torch.ones(B).to(device)
    z = torch.randn_like(x_t).to(device)
    dw = abs(dt)**(1/2) * z
    g = model.sde.diffusion(t, x_t).to(device)

    sig_t = model.sde.sigma(t).to(device).unsqueeze(1).unsqueeze(2).unsqueeze(3)

    score_likelihood = - (y - x_t) / (sig_t ** 2 + eta**2)
    score_prior = model.score(t, x_t).to(device)

    dx = - g**2 * ( score_likelihood + score_prior) * dt + g * dw

    x_t += dx
    

torch.save(x_t, f'{script_dir}/images/samples_unc.pt')
# %%
