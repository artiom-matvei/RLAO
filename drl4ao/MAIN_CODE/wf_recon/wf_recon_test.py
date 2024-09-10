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

# %%
LE_PSFs = []
SE_PSFs = []
SRs = []
rewards = []
accu_reward = 0

obs = torch.tensor(env.reset_soft_wfs()).float()

network = Reconstructor(1,1,11, env.xvalid, env.yvalid)
network.load_state_dict(torch.load(savedir+'/reconstructor.pt', map_location=torch.device('cpu')))
network.eval()

#%%
for i in range(args.nLoop):
    a=time.time()
    # print(env.gainCL)
    obs = torch.tensor(obs).clone().detach.float().unsqueeze(0).unsqueeze(0)
    with torch.no_grad(): 
        action = - env.gainCL * network(obs).squeeze(0).squeeze(0) #env.integrator()
    obs, reward,strehl, done, info = env.step_wfs(i,action*1e-6)  
    accu_reward+= reward

    b= time.time()
    print('Elapsed time: ' + str(b-a) +' s')
    # LE_PSF, SE_PSF = env.render(i)
    # LE_PSF, SE_PSF = env.render4plot(i)
    # env.render4plot(i)

    print('Loop '+str(i+1)+'/'+str(args.nLoop)+' Gain: '+str(env.gainCL)+' Turbulence: '+str(env.total[i])+' -- Residual:' +str(env.residual[i])+ '\n')
    print("SR: " +str(strehl))
    if (i+1) % 500 == 0:
        sr = env.calculate_strehl_AVG()
        SRs.append(sr)
        rewards.append(accu_reward)
        accu_reward = 0


print(SRs)
print(rewards)
print("Saving Data")
save_plots(savedir,SRs,rewards,env.LE_PSF) #savedir,evals,reward_sums,env.LE_PS
# %%

fig, ax = plt.subplots(1,2, figsize=(10,5))

env.atm.generateNewPhaseScreen(31)
env.tel*env.wfs

obs = torch.tensor(env.wfs.cam.frame).float().unsqueeze(0).unsqueeze(0)

cax1 = ax[0].imshow(env.OPD_on_dm() * env.tel.pupil)
cax2 = ax[1].imshow(OPD_model(network(obs), modes, res).squeeze(0).squeeze(0).detach().numpy()*env.tel.pupil)

ax[0].axis('off')
ax[1].axis('off')


ax[0].set_title('Random Phase', size=15)
ax[1].set_title('Reconstructed Phase', size=15)

plt.colorbar(cax1, ax=ax[0])
plt.colorbar(cax2, ax=ax[1])

plt.tight_layout()

plt.show()

# %%
