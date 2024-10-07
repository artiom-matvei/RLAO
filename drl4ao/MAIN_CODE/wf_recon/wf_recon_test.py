#%%
import os,sys
import torch
import torch.optim as optim
import torch.nn as nn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import numpy as np
# from PO4AO.mbrl_funcsRAZOR import get_env
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

# SimpleNamespace takes a dict and allows the use of
# keys as attributes. ex: args['r0'] -> args.r0
# try:
#     args = SimpleNamespace(**read_yaml_file('./Conf/razor_config_po4ao.yaml'))
# except:
#     args = SimpleNamespace(**read_yaml_file('../Conf/razor_config_po4ao.yaml'))
try:
    args = SimpleNamespace(**read_yaml_file('./Conf/papyrus_config.yaml'))
except:
    args = SimpleNamespace(**read_yaml_file('../Conf/papyrus_config.yaml'))

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
#                         min_scale=1e-6, max_scale=1e-6)

data_dir_path = '/home/parker09/projects/def-lplevass/parker09/RLAO/drl4ao/MAIN_CODE/wf_recon/datasets/atm_stats'

ds_size = 300000


# %%
# Set the random seed for reproducibility
np.random.seed(432574358)

# Shuffle the data indices
indices = np.arange(ds_size)
np.random.shuffle(indices)

# Define the split ratios
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

# Calculate the split indices
train_size = int(train_ratio * ds_size)
val_size = int(val_ratio * ds_size)

# Get the corresponding indices for each set
train_indices = indices[:train_size]
val_indices = indices[train_size:train_size + val_size]
test_indices = indices[train_size + val_size:]

input_file_path = data_dir_path+'/wfs_frames_unmodulated_300.npy'
target_file_path = data_dir_path+'/dm_cmds_unmodulated_300.npy'

dm_shape = env.dm.coefs.shape
wfs_shape=env.wfs.cam.frame.shape

D_val = FileDataset(input_file_path, target_file_path, val_indices, dm_shape=dm_shape, wfs_shape=wfs_shape, size=300000)

#%%
wfsf = torch.zeros((3,4,24,24))
dm_c = torch.zeros(3,1,357)

for i in range(3):
    wfsf[i], dm_c[i] = D_val.__getitem__(np.random.randint(0,20000))


# %%

# try:
#     reconstructor = load_model(savedir+'/best_model_OL.pt')

# except:
checkpoint = torch.load(savedir+'/models/tmp/unmod.pt',map_location=device)


# Make sure to use the correct network before loading the state dict
reconstructor = Unet_big(env.xvalid,env.yvalid)
# Restore the regular model and optimizer state
reconstructor.load_state_dict(checkpoint['model_state_dict'])

#%% 
# Load the model
# network = load_model(savedir+'/reconstructor_cmd_asinh.pt')
reconstructor.eval()
# Make predictions
# obs = torch.tensor(wfsf).float().unsqueeze(1)




# mean = obs.mean(dim=[2, 3], keepdim=True)  # Mean over height and width dimensions
# std = obs.std(dim=[2, 3], keepdim=True)    # Std over height and width dimensions

# Perform mean subtraction and scaling by standard deviation
# normalized_tensor = (obs - mean) / std

# reshaped_input = obs.view(-1, 2, 24, 2, 24).permute(0, 1, 3, 2, 4).contiguous().view(-1, 4, 24, 24)

with torch.no_grad():
    pred = reconstructor(wfsf)
    # pred_OPD = OPD_model(network(obs), modes, res)


# Run ground truth commands through the model

# Reshape commands into image
cmd_img = np.array([env.vec_to_img(torch.tensor(i).float()) for i in dm_c])

# gt_opd = OPD_model(torch.tensor(cmd_img).unsqueeze(1), modes, res)


#%%
fig, ax = plt.subplots(4,3, figsize=(10,13))

vrange = .3

for i in range(3):
    top = torch.cat([wfsf[i][0], wfsf[i][1]], dim=1)  # Concatenate top wfsf i][along width
    bottom = torch.cat([wfsf[i][2], wfsf[i][3]], dim=1)  # Concatenate bottom quarters along width

    # Step 3: Concatenate the top and bottom parts along the height
    stitched_image = torch.cat([top, bottom], dim=0)

    cax1 = ax[0,i].imshow(stitched_image.detach().numpy())
    cax2 = ax[1,i].imshow(pred[i].squeeze(0).detach().numpy(), vmin=-vrange, vmax=vrange)
    cax3 = ax[2,i].imshow(cmd_img[i] , vmin=-vrange, vmax=vrange)
    cax4 = ax[3,i].imshow(cmd_img[i]  - pred[i].squeeze(0).detach().numpy(),\
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

plt.style.use('ggplot')

# plot losses
tag = 'unmod'

loss_dir = savedir+ '/losses'
train_loss = np.load(loss_dir+ '/train_loss_' + tag + '.npy')
val_loss = np.load(loss_dir + '/val_loss_' + tag + '.npy')
# ema_loss = np.load(loss_dir + '/ema_val_loss_' + tag + '.npy')


plt.plot(train_loss, label='Train Loss', color='k')
plt.plot(val_loss, label='Val Loss', color='r')
# plt.plot(ema_loss, label='ema loss', ls='--', c='k')

plt.axvline(np.argmin(val_loss), color='k', ls='--', alpha=0.4, label='Best Model')

# plt.axhline(0.0387)

plt.yscale('log')
plt.legend()

# plt.xlim(80, 100)
# plt.ylim(0.03, 0.05)

plt.title('Training Loss of Wavefront Reconstructor')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')

plt.show()
# %%
### Closed loop test ### 

reconstructor.eval()
c_int = 1.
c_net = 1. - c_int

env.atm.generateNewPhaseScreen(133)
env.dm.coefs = 0

env.tel*env.dm*env.wfs

LE_PSFs= []
SE_PSFs = []
SRs = []
SR_std = []
rewards = []
accu_reward = 0

slope = env.reset_soft()

wfsf = env.wfs.cam.frame.copy()

obs = torch.tensor(wfsf).float().unsqueeze(1)

args.nLoop = 500

for i in range(args.nLoop):
    # reshaped_input = obs.view(-1, 2, 24, 2, 24).permute(0, 1, 3,2, 4).contiguous().view(-1, 4, 24, 24)

    # with torch.no_grad():
    #     action = -1 * (- c_int * 0.9 * slope + c_net * np.sinh(reconstructor(reshaped_input).squeeze()))

    action = 0.9 * slope

    a=time.time()
    # print(env.gainCL)
    # action = env.gainCL * obs #env.integrator()
    slope, reward,strehl, done, info = env.step(i,action)  

    # obs = torch.tensor(env.wfs.cam.frame.copy()).float().unsqueeze(1)

    accu_reward+= reward

    b= time.time()
    # print('Elapsed time: ' + str(b-a) +' s')
    # LE_PSF, SE_PSF = env.render(i)
    # LE_PSF, SE_PSF = env.render4plot(i)
    # env.render4plot(i)
    # if (i+1) % 100 == 0:
    #     sr, std = env.calculate_strehl_AVG()
    #     SRs.append(sr)
    #     SR_std.append(std)
    #     rewards.append(accu_reward)
    #     accu_reward = 0
    #     print(f'\n Frame {i} \n')

    SRs.append(strehl)


    # print('Loop '+str(i+1)+'/'+str(args.nLoop)+' Gain: '+str(env.gainCL)+' Turbulence: '+str(env.total[i])+' -- Residual:' +str(env.residual[i])+ '\n')
    print(f"Iter {i+1}, SR: {strehl * 100:.1f}%")
# %%
