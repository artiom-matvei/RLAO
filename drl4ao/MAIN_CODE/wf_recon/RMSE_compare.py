#%%
import os,sys
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import numpy as np
from PO4AO.mbrl import get_env
from ML_stuff.dataset_tools import ImageDataset, FileDataset, make_diverse_dataset, read_yaml_file
from ML_stuff.models import Reconstructor, Reconstructor_2, build_unet, Unet_big
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

try:
    args = SimpleNamespace(**read_yaml_file('./Conf/papyrus_config.yaml'))
except:
    args = SimpleNamespace(**read_yaml_file('../Conf/papyrus_config.yaml'))

savedir = os.path.dirname(__file__)
env = get_env(args)

#%%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

size = int(1e3)

dm_commands = np.zeros(shape=(size, *env.dm.coefs.shape), dtype='float32')
wfs_frames = np.zeros(shape=(size, *env.wfs.cam.frame.shape), dtype='float32')
wfs_frames_filtered = np.zeros(shape=(size, *env.wfs.cam.frame.shape), dtype='float32')

m2opd = np.load(os.path.dirname(__file__)+'/M2OPD_500modes.npy')
opd2m = np.linalg.pinv(m2opd)
m2c   = np.load(os.path.dirname(__file__)+'/M2C_357modes.npy')
xpupil, ypupil = np.where(env.tel.pupil == 1)

for j in range(size):
    
    # Random Atmospheric Phase Screen
    env.atm.generateNewPhaseScreen(np.random.randint(0, 1e8))
    env.tel*env.wfs

    # Save the WFS frame
    wfs_frames[j] = np.float32(env.wfs.cam.frame.copy())

    # Zernike Modes of Phase
    phase = env.tel.OPD.copy()
    modes = opd2m@phase[xpupil, ypupil]

    # Save nModes controllable modes on DM as Grount Truth Commands
    nModes = 30
    dm_commands[j] = np.float32(m2c[:, :nModes]@modes[:nModes])

    # Filter Phase to nModes modes
    filteredPhase = np.zeros_like(phase)
    filteredPhase[xpupil, ypupil] = m2opd[:,:nModes]@modes[:nModes]

    # Propagate Filtered Phase to WFS
    env.tel.OPD = filteredPhase.copy()
    env.tel*env.wfs

    # Save the WFS frame
    wfs_frames_filtered[j] = np.float32(env.wfs.cam.frame.copy())

# %%
checkpoint_filt = torch.load(savedir+'/models/thesis_models/filt_atmos.pt',map_location=device)
# Make sure to use the correct network before loading the state dict
reconstructor_filt = Unet_big(env.xvalid,env.yvalid)
# Restore the regular model and optimizer state
reconstructor_filt.load_state_dict(checkpoint_filt['model_state_dict'])
reconstructor_filt.eval()

with torch.no_grad():
    reshaped_wfs = torch.from_numpy(wfs_frames).view(1000, 2, 24, 2, 24).permute(0,1, 3, 2, 4).contiguous().view(1000, 4, 24, 24)
    pred_filt = reconstructor_filt(reshaped_wfs).squeeze(1)

checkpoint_full = torch.load(savedir+'/models/thesis_models/atmos.pt',map_location=device)
reconstructor_full = Unet_big(env.xvalid,env.yvalid)
# Restore the regular model and optimizer state
reconstructor_full.load_state_dict(checkpoint_full['model_state_dict'])
reconstructor_full.eval()

with torch.no_grad():
    reshaped_wfs = torch.from_numpy(wfs_frames).view(1000, 2, 24, 2, 24).permute(0,1, 3, 2, 4).contiguous().view(1000, 4, 24, 24)
    pred_full = reconstructor_full(reshaped_wfs).squeeze(1)



lin_pred = np.zeros(shape=(size, *env.dm.coefs.shape), dtype='float32')
for i in range(size):
    env.wfs.cam.frame = wfs_frames[i]
    _, signal = env.wfs.signalProcessing()

    lin_pred[i] = np.sinh(env.reconstructor@signal) * 1e6


cmd_img = np.sinh(np.array([env.vec_to_img(torch.tensor(i).float()) for i in dm_commands])) * 1e6

#%%
fig, ax = plt.subplots(4,3, figsize=(10,13))
vrange = .3
for i in range(3):

    cax1 = ax[0,i].imshow(wfs_frames[i])
    cax2 = ax[1,i].imshow(pred_filt[i].squeeze(0).detach().numpy(), vmin=-vrange, vmax=vrange)
    cax3 = ax[2,i].imshow(cmd_img[i] , vmin=-vrange, vmax=vrange)
    cax4 = ax[3,i].imshow(cmd_img[i]  - pred_filt[i].squeeze(0).detach().numpy(),\
                                                                 vmin=-vrange, vmax=vrange)
    ax[0,i].axis('off')
    ax[1,i].axis('off')
    ax[2,i].axis('off')
    ax[3,i].axis('off')

    ax[0,i].set_title('Input WFS Image', size=15)
    ax[1,i].set_title('Reconstructed Phase', size=15)
    ax[2,i].set_title('Ground Truth Phase', size=15)
    ax[3,i].set_title('Difference', size=15)

plt.show()

#%%

def vec(imgs):
    np.zeros(shape=(size, *env.dm.coefs.shape), dtype='float32')

    vec = imgs[:, env.xvalid, env.yvalid]
    return vec

rmse_lin = np.sqrt(np.mean((np.sinh(dm_commands)*1e6  - lin_pred)**2, axis=1))
rmse_net_filt = np.sqrt(np.mean((np.sinh(dm_commands)*1e6  - vec(pred_filt).detach().numpy())**2, axis=1))
rmse_net_full = np.sqrt(np.mean((np.sinh(dm_commands)*1e6  - vec(pred_full).detach().numpy())**2, axis=1))


plt.figure(figsize=(6, 7))
plt.boxplot([rmse_lin,rmse_net_filt, rmse_net_full], labels=['Linear', "Network (Filtered)", "Network (Full)"])
plt.ylabel("RMSE")
plt.title("RMS Reconstruction Error")
plt.grid(axis='y', linestyle='--', alpha=0.7)
# plt.yscale('log')

plt.show()

# %%

