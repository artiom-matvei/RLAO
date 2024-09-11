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
from PO4AO.conv_models_simple import Reconstructor, ImageDataset
from types import SimpleNamespace
import matplotlib.pyplot as plt
# Customize rcParams for specific adjustments
plt.rcParams['image.cmap'] = 'inferno'      # Set colormap to inferno for imshow
plt.rcParams['figure.facecolor'] = 'black'  # Set figure background to black
plt.rcParams['axes.facecolor'] = 'black'    # Set axes background to black
plt.rcParams['savefig.facecolor'] = 'black' # Set saved figures' background to black
plt.rcParams['axes.grid'] = False           # Disable grid for a cleaner look on dark background
plt.rcParams['xtick.color'] = 'white'
plt.rcParams['ytick.color'] = 'white'
plt.rcParams['axes.labelcolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'white'
plt.rcParams['legend.facecolor'] = 'gray'

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
    model = Reconstructor(1,1,11, env.xvalid,env.yvalid)  # Replace with your model class
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # Set the model to evaluation mode
    return model



# %%
# Make some fresh data
# wfsf, dmc = make_diverse_dataset(env, size=1, num_scale=3,\
#                         min_scale=1e-9, max_scale=1e-6)

X = np.load(savedir+'/wfs_frames.npy', mmap_mode='r')
y = np.load(savedir+'/dm_cmds.npy', mmap_mode='r')

np.random.seed(24)

# Shuffle the data indices
indices = np.arange(10000)
np.random.shuffle(indices)

# Define the split ratios
train_ratio = 0.6
val_ratio = 0.2
test_ratio = 0.2

# Calculate the split indices
train_size = int(train_ratio * 10000)
val_size = int(val_ratio * 10000)

# Get the corresponding indices for each set
train_indices = indices[:train_size]
val_indices = indices[train_size:train_size + val_size]
test_indices = indices[train_size + val_size:]

# Split the data
X_train, X_val, X_test = X[train_indices], X[val_indices], X[test_indices]
y_train, y_val, y_test = y[train_indices], y[val_indices], y[test_indices]

# Now you have:
# X_train, y_train: training set
# X_val, y_val: validation set
# X_test, y_test: test set
# D_train = ImageDataset(X_train, y_train)
# D_test = ImageDataset(X_test, y_test)
D_val = ImageDataset(X_val, y_val)


# %%

reconstructor = load_model(savedir+'/reconstructor_cmd.pt')
optimizer = optim.Adam(reconstructor.parameters(), lr=0.00001)
criterion = nn.SmoothL1Loss()

reconstructor.to(device)

# train_loader = DataLoader(D_train, batch_size=32, shuffle=True)
val_loader = DataLoader(D_val, batch_size=32, shuffle=True)
# test_loader = DataLoader(D_test, batch_size=32, shuffle=True)


train_losses = []
val_losses = []

n_epochs = 1
for epoch in range(n_epochs):
    reconstructor.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            #forward pass
            outputs = env.img_to_vec(reconstructor(inputs))


            loss = criterion(outputs, targets)
            val_loss += loss.item()

    avg_val_loss = val_loss/len(val_loader)
    val_losses.append(avg_val_loss)

    # with open("training_progress.txt", "a") as f:  # 'a' mode appends to the file
    #     f.write(f"Epoch {epoch + 1}/{n_epochs}, Loss: {avg_val_loss}\n")

    print(f'Epoch {epoch+1}/{n_epochs}, Validation Loss: {avg_val_loss}')



#%% 
# Load the model
network = load_model(savedir+'/reconstructor_cmd_asinh.pt')

# Make predictions
obs = torch.tensor(wfsf).float().unsqueeze(1)

with torch.no_grad():
    pred = network(obs)
    # pred_OPD = OPD_model(network(obs), modes, res)


# Run ground truth commands through the model

# Reshape commands into image
cmd_img = np.array([env.vec_to_img(torch.tensor(i).float()) for i in dmc])

# gt_opd = OPD_model(torch.tensor(cmd_img).unsqueeze(1), modes, res)


#%%
fig, ax = plt.subplots(4,3, figsize=(10,13))

for i in range(3):
    cax1 = ax[0,i].imshow(wfsf[i])
    cax2 = ax[1,i].imshow(pred[i].squeeze(0).detach().numpy(), vmin=-15, vmax=15)
    cax3 = ax[2,i].imshow(np.arcsinh(cmd_img[i] / 1e-9), vmin=-15, vmax=15)
    cax4 = ax[3,i].imshow(np.arcsinh(cmd_img[i] / 1e-9) - pred[i].squeeze(0).detach().numpy(),\
                                                                 vmin=-15, vmax=15)


    ax[0,i].axis('off')
    ax[1,i].axis('off')
    ax[2,i].axis('off')
    ax[3,i].axis('off')

    ax[0,i].set_title('Input WFS Image', size=15)
    ax[1,i].set_title('Reconstructed Phase', size=15)
    ax[2,i].set_title('Ground Truth Phase', size=15)
    ax[3,i].set_title('Difference', size=15)

# plt.tight_layout()

plt.show()

# %%
