#%%
import os,sys
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import pickle


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from PO4AO.util_simple import read_yaml_file, append_to_pickle_file #TorchWrapper,

import time
import numpy as np
from PO4AO.mbrl_funcsRAZOR import get_env, make_diverse_dataset
from PO4AO.conv_models_simple import Reconstructor
from types import SimpleNamespace
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'inferno' 
# SimpleNamespace takes a dict and allows the use of
# keys as attributes. ex: args['r0'] -> args.r0
try:
    args = SimpleNamespace(**read_yaml_file('./Conf/razor_config_po4ao.yaml'))
except:
    args = SimpleNamespace(**read_yaml_file('../Conf/razor_config_po4ao.yaml'))

savedir = os.path.dirname(__file__)
#%%
env = get_env(args)
#%%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

modes = torch.tensor(env.dm.modes.copy()).to(device).float()
res = env.dm.resolution
# tel_mask = torch.tensor(env.tel.pupil.copy()).to(device)

def OPD_model(dm_cmd, modes, res):

    vec_cmd = dm_cmd[:,:,env.xvalid, env.yvalid]
    dm_opd = torch.matmul(vec_cmd,modes.unsqueeze(0).unsqueeze(0).transpose(-1,-2)).squeeze(0)

    dm_opd = torch.reshape(dm_opd, (-1,res,res)).unsqueeze(1)

    return dm_opd

def load_model(model_path):
    """Loads the trained model from a state dict."""
    model = Reconstructor()  # Replace with your model class
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode
    return model



# %%
# Make some fresh data
wfsf, dmc = make_diverse_dataset(env, size=1, num_scale=3,\
                        min_scale=1e-9, max_scale=1e-6)

# Load the model
network = load_model(savedir+'/reconstructor_cmd.pt')

# Make predictions
obs = torch.tensor(wfsf).float().unsqueeze(1).unsqueeze(1)
pred = OPD_model(network(obs), modes, res)

# Run ground truth commands through the model
gt = OPD_model(torch.tensor(dmc).float().unsqueeze(0).unsqueeze(0), modes, res)

# %%

fig, ax = plt.subplots(2,3, figsize=(15,5))

for i in range(3):
    cax1 = ax[0,i].imshow(wfsf[i])
    cax2 = ax[1,i].imshow(pred.squeeze(0).squeeze(0).detach().numpy()[i]*env.tel.pupil)
    cax3 = ax[2,i].imshow(gt.squeeze(0).squeeze(0).detach().numpy()[i]*env.tel.pupil)


    ax[0,i].axis('off')
    ax[1,i].axis('off')
    ax[2,i].axis('off')


    ax[0,i].set_title('Input WFS Image', size=15)
    ax[1,i].set_title('Reconstructed Phase', size=15)
    ax[2,i].set_title('Ground Truth Phase', size=15)
    

plt.colorbar(cax1, ax=ax[0])
plt.colorbar(cax2, ax=ax[1])

plt.tight_layout()

plt.show()

# %%
