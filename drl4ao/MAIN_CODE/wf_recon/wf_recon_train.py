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
from PO4AO.mbrl_funcsRAZOR import get_env, make_diverse_dataset
from PO4AO.conv_models_simple import Reconstructor, ImageDataset
from Plots.plots import save_plots
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

env = get_env(args)

# Generate the dataset of wfs images and phase maps
env.tel.resetOPD()
env.tel*env.dm*env.wfs

# plt.imshow(env.tel.OPD)
# plt.show()
# plt.imshow(env.dm.OPD)
# plt.show()
# plt.imshow(env.wfs.cam.frame)

# %%

# wfsf, dmc = make_diverse_dataset(env, size=10000, num_scale=1,\
#                      min_scale=1e-6, max_scale=1e-6)

# # Save the dataset
# np.save(savedir+'/datasets/wfs_frames_emin6', wfsf)
# np.save(savedir+'/datasets/dm_cmds_emin6', dmc)

#%%
#CHECK DATA FROM THE DATASET
# print(len(wfsf), len(dmc))

# frame = 21

# plt.imshow(wfsf[frame])
# plt.show()

# plt.imshow(env.vec_to_img(torch.tensor(dmc[frame]).float()).detach().numpy(),\
#                     vmin=-1e-6, vmax=1e-6)
# plt.colorbar()
# plt.show()

# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %%

X = np.load(savedir+'/datasets/wfs_frames_emin6.npy')
y_raw = np.load(savedir+'/datasets/dm_cmds_emin6.npy')

#Transform commands to regular scale
# X = wfsf
# y_raw = dmc
y = np.arcsinh(y_raw / 1e-6)


# Set the random seed for reproducibility
np.random.seed(432)

# Shuffle the data indices
indices = np.arange(len(X))
np.random.shuffle(indices)

# Define the split ratios
train_ratio = 0.6
val_ratio = 0.2
test_ratio = 0.2

# Calculate the split indices
train_size = int(train_ratio * len(X))
val_size = int(val_ratio * len(X))

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
D_train = ImageDataset(X_train, y_train)
D_test = ImageDataset(X_test, y_test)
D_val = ImageDataset(X_val, y_val)


# %%

reconstructor = Reconstructor(1,1,11, env.xvalid, env.yvalid)

# EMA of model parameters
ema_reconstructor = torch.optim.swa_utils.AveragedModel(reconstructor, \
    multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999))

optimizer = optim.Adam(reconstructor.parameters(), lr=0.0001)
criterion = nn.MSELoss()

reconstructor.to(device)
ema_reconstructor.to(device)


train_loader = DataLoader(D_train, batch_size=32, shuffle=True)
val_loader = DataLoader(D_val, batch_size=32, shuffle=True)
test_loader = DataLoader(D_test, batch_size=32, shuffle=True)


train_losses = []
val_losses = []
ema_val_losses = []
# Variable to store the best validation loss and path to save the model
best_val_loss = float('inf')  # Initialize to infinity
save_path = savedir+'/models/best_models_ema_OL.pt'  # Path to save the best model

n_epochs = 300
for epoch in range(n_epochs):

    #Training phase
    reconstructor.train()
    running_loss = 0.0
    for inputs, targets in train_loader:
        # Zero the parameter gradients
        optimizer.zero_grad()
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # Forward pass
        outputs = env.img_to_vec(reconstructor(inputs))

        # Get the OPD from the model
        loss = criterion(outputs, targets)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        ema_reconstructor.update_parameters(reconstructor)

        running_loss += loss.item()

    
    avg_train_loss = running_loss/len(train_loader)
    train_losses.append(avg_train_loss)
    print(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_train_loss}")


    # Validation phase
    reconstructor.eval()
    val_loss = 0.0
    ema_val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            #forward pass
            outputs = env.img_to_vec(reconstructor(inputs))
            ema_outputs = env.img_to_vec(ema_reconstructor(inputs))


            loss = criterion(outputs, targets)
            val_loss += loss.item()

            ema_loss = criterion(ema_outputs, targets)
            ema_val_loss += ema_loss.item()

    avg_val_loss = val_loss/len(val_loader)
    val_losses.append(avg_val_loss)

    avg_ema_val_loss = ema_val_loss/len(val_loader)
    ema_val_losses.append(avg_ema_val_loss)



    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss  # Update the best validation loss
        print(f"Validation loss improved to {avg_val_loss:.4f}, saving model...")
        
        # Save the best model
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': reconstructor.state_dict(),
            'ema_model_state_dict': ema_reconstructor.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': best_val_loss,
        }, save_path)

    with open("training_progress.txt", "a") as f:  # 'a' mode appends to the file
        f.write(f"Epoch {epoch + 1}/{n_epochs}, Loss: {avg_val_loss}\n")

    print(f'Epoch {epoch+1}/{n_epochs}, Validation Loss: {avg_val_loss}')

    np.save(savedir+'/losses/train_loss_ema', train_losses)
    np.save(savedir+'/losses/val_loss_ema', val_losses)
    np.save(savedir+'/losses/ema_val_loss_ema', ema_val_losses)

# Test phase (after all epochs)
reconstructor.eval()  # Set the model to evaluation mode
test_loss = 0.0
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Forward pass
        outputs = env.img_to_vec(reconstructor(inputs))

        # Compute loss
        loss = criterion(outputs, targets)

        test_loss += loss.item()

# Calculate average test loss
avg_test_loss = test_loss / len(test_loader)
print(f"Test Loss: {avg_test_loss}")


np.save(savedir+'/losses/train_loss_ema', train_losses)
np.save(savedir+'/losses/val_loss_ema', val_losses)
np.save(savedir+'/losses/ema_val_loss_ema', ema_val_losses)
torch.save(reconstructor.state_dict(), savedir+'/models/last_ema.pt')

# %%
