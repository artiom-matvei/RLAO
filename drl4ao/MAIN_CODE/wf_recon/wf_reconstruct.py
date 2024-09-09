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
from PO4AO.util_simple import read_yaml_file #TorchWrapper, 

import time
import numpy as np
from PO4AO.mbrl_funcsRAZOR import get_env, get_phase_dataset, train_reconstructor
from PO4AO.conv_models_simple import Reconstructor, ImageDataset
from types import SimpleNamespace
import matplotlib.pyplot as plt
# SimpleNamespace takes a dict and allows the use of
# keys as attributes. ex: args['r0'] -> args.r0
args = SimpleNamespace(**read_yaml_file('../Conf/razor_config_po4ao.yaml'))

#%%
env = get_env(args)
#%%

# Generate the dataset of wfs images and phase maps
phase, dataset = get_phase_dataset(env, 2048)

dataset.to_pickle('phase_dataset.pkl')


# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

modes = torch.tensor(env.dm.modes.copy()).to(device).float()
res = env.dm.resolution
# tel_mask = torch.tensor(env.tel.pupil.copy()).to(device)

def OPD_model(dm_cmd, modes, res):

    vec_cmd = dm_cmd[:,:,env.xvalid, env.yvalid]
    dm_opd = torch.matmul(vec_cmd,modes.unsqueeze(0).unsqueeze(0).transpose(-1,-2)).squeeze(0)

    dm_opd = torch.reshape(dm_opd, (-1,res,res)).unsqueeze(1)

    return dm_opd

# %%
savedir = os.path.dirname(__file__)

dataset_raw = pd.read_pickle(savedir+'phase_dataset.pkl').reset_index(drop=True)
ds_torch = ImageDataset(dataset_raw, 'wfs', 'dm')

# %%

reconstructor = Reconstructor(1,1,11, env.xvalid, env.yvalid)
optimizer = optim.Adam(reconstructor.parameters(), lr=0.001)
criterion = nn.SmoothL1Loss()

reconstructor.train()
reconstructor.to(device)

dataloader = DataLoader(ds_torch, batch_size=32, shuffle=True)


n_epochs = 10
for epoch in range(n_epochs):
    running_loss = 0.0
    for inputs, targets in dataloader:
        # Zero the parameter gradients
        optimizer.zero_grad()
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # Forward pass
        outputs = reconstructor(inputs)

        # Get the OPD from the model
        dm_OPD = OPD_model(outputs, modes, res)

        loss = criterion(dm_OPD, targets)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{n_epochs}, Loss: {running_loss/len(dataloader)}")



torch.save(reconstructor.state_dict(), savedir+'reconstructor.pt')
# %%
