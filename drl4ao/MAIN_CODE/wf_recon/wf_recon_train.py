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

# Generate the dataset of wfs images and phase maps
dataset = make_diverse_dataset(env, size=10, num_scale=20)

dataset.to_pickle(savedir+'/diverse_phases_test.pkl')


# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %%

dataset_raw = pd.read_pickle(savedir+'/diverse_phases_test.pkl').reset_index(drop=True)

X = dataset_raw['wfs']
y = dataset_raw['dm']

# Set the random seed for reproducibility
np.random.seed(42)

# Shuffle the data indices
indices = np.arange(len(dataset_raw))
np.random.shuffle(indices)

# Define the split ratios
train_ratio = 0.6
val_ratio = 0.2
test_ratio = 0.2

# Calculate the split indices
train_size = int(train_ratio * len(dataset_raw))
val_size = int(val_ratio * len(dataset_raw))

# Get the corresponding indices for each set
train_indices = indices[:train_size]
val_indices = indices[train_size:train_size + val_size]
test_indices = indices[train_size + val_size:]

# Split the data
X_train, X_val, X_test = X.iloc[train_indices], X.iloc[val_indices], X.iloc[test_indices]
y_train, y_val, y_test = y.iloc[train_indices], y.iloc[val_indices], y.iloc[test_indices]

df_train = pd.DataFrame({'wfs': X_train, 'dm': y_train})
df_test = pd.DataFrame({'wfs': X_test, 'dm': y_test})
df_val = pd.DataFrame({'wfs': X_val, 'dm': y_val})

# Now you have:
# X_train, y_train: training set
# X_val, y_val: validation set
# X_test, y_test: test set
D_train = ImageDataset(df_train, 'wfs', 'dm')
D_test = ImageDataset(df_test, 'wfs', 'dm')
D_val = ImageDataset(df_val, 'wfs', 'dm')


# %%

reconstructor = Reconstructor(1,1,11, env.xvalid, env.yvalid)
optimizer = optim.Adam(reconstructor.parameters(), lr=0.001)
criterion = nn.SmoothL1Loss()

reconstructor.to(device)

train_loader = DataLoader(D_train, batch_size=32, shuffle=True)
val_loader = DataLoader(D_val, batch_size=32, shuffle=True)
test_loader = DataLoader(D_test, batch_size=32, shuffle=True)


train_losses = []
val_losses = []

n_epochs = 150
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

        running_loss += loss.item()

    
    avg_train_loss = running_loss/len(train_loader)
    train_losses.append(avg_train_loss)
    print(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_train_loss:.4f}")


    # Validation phase
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
    print(f'Epoch {epoch+1}/{n_epochs}, Validation Loss: {avg_val_loss:.4f}')

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
print(f"Test Loss: {avg_test_loss:.4f}")


np.save(savedir+'/train_loss', train_loss)
np.save(savedir+'/val_loss', val_loss)
torch.save(reconstructor.state_dict(), savedir+'/reconstructor_cmd.pt')


# %%
