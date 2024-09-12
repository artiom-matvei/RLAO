#%%
import os,sys
import torch
import torch.optim as optim
import torch.nn as nn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import numpy as np
from PO4AO.mbrl_funcsRAZOR import get_env
from ML_stuff.dataset_tools import ImageDataset, FileDataset, make_diverse_dataset, read_yaml_file
from ML_stuff.models import Reconstructor, Reconstructor_2
from types import SimpleNamespace
import matplotlib.pyplot as plt
# Customize rcParams for specific adjustments
plt.rcParams['image.cmap'] = 'inferno'      # Set colormap to inferno for imshow
plt.rcParams['figure.facecolor'] = 'white'  # Set figure background to black
plt.rcParams['axes.facecolor'] = 'white'    # Set axes background to black
plt.rcParams['savefig.facecolor'] = 'white' # Set saved figures' background to black
plt.rcParams['axes.grid'] = False           # Disable grid for a cleaner look on dark background
plt.rcParams['xtick.color'] = 'black'
plt.rcParams['ytick.color'] = 'black'
plt.rcParams['axes.labelcolor'] = 'black'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['legend.facecolor'] = 'white'

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
wfsf, dmc = make_diverse_dataset(env, size=1, num_scale=3,\
                        min_scale=1e-6, max_scale=1e-6)

# %%

# try:
#     reconstructor = load_model(savedir+'/best_model_OL.pt')

# except:
checkpoint = torch.load(savedir+'/best_model_OL.pt',map_location=device)


# Make sure to use the correct network before loading the state dict
reconstructor = Reconstructor(1,1,11, env.xvalid,env.yvalid)
# Restore the regular model and optimizer state
reconstructor.load_state_dict(checkpoint['model_state_dict'])

#%% 
# Load the model
# network = load_model(savedir+'/reconstructor_cmd_asinh.pt')
reconstructor.eval()
# Make predictions
obs = torch.tensor(wfsf).float().unsqueeze(1)

with torch.no_grad():
    pred = reconstructor(obs)
    # pred_OPD = OPD_model(network(obs), modes, res)


# Run ground truth commands through the model

# Reshape commands into image
cmd_img = np.array([env.vec_to_img(torch.tensor(i).float()) for i in dmc])

# gt_opd = OPD_model(torch.tensor(cmd_img).unsqueeze(1), modes, res)


#%%
fig, ax = plt.subplots(4,3, figsize=(10,13))

vrange = 3

for i in range(3):
    cax1 = ax[0,i].imshow(wfsf[i])
    cax2 = ax[1,i].imshow(pred[i].squeeze(0).detach().numpy(), vmin=-vrange, vmax=vrange)
    cax3 = ax[2,i].imshow(np.arcsinh(cmd_img[i] / 1e-6), vmin=-vrange, vmax=vrange)
    cax4 = ax[3,i].imshow(np.arcsinh(cmd_img[i] / 1e-6) - pred[i].squeeze(0).detach().numpy(),\
                                                                 vmin=-vrange, vmax=vrange)


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

# plot losses
tag = 'ema'
loss_dir = savedir+ '/losses'
train_loss = np.load(loss_dir+ '/train_loss_' + tag + '.npy')
val_loss = np.load(loss_dir + '/val_loss_' + tag + '.npy')
ema_loss = np.load(loss_dir + '/ema_val_loss_' + tag + '.npy')

plt.plot(train_loss, label='Train Loss', color='k')
plt.plot(val_loss, label='Val Loss', color='r')
# plt.plot(ema_loss, label='ema loss', ls='--', c='k')

plt.axvline(np.argmin(val_loss), color='k', ls='--', alpha=0.4, label='Best Model')

plt.yscale('log')
plt.legend()

plt.title('Training Loss of Wavefront Reconstructor')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')

plt.show()
# %%
