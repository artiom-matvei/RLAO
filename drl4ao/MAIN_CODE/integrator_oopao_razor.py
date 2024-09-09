"""
OOPAO module for the integrator
@author: Raissa Camelo (LAM) git: @srtacamelo
"""

#%%
import os,sys
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

# from parser_Configurations import Config, ConfigAction
# from OOPAOEnv.OOPAOEnvRazor import OOPAO
from PO4AO.util_simple import read_yaml_file #TorchWrapper, 
from Plots.plots import save_plots
# import matplotlib.pyplot as plt
# import argparse
import time
import numpy as np
from PO4AO.mbrl_funcsRAZOR import get_env, get_phase_dataset
from types import SimpleNamespace
import matplotlib.pyplot as plt
# SimpleNamespace takes a dict and allows the use of
# keys as attributes. ex: args['r0'] -> args.r0
args = SimpleNamespace(**read_yaml_file('Conf/razor_config_po4ao.yaml'))

#%%

env = get_env(args)

for gainCL in args.gain_list:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    savedir = '../../logs/'+args.savedir+'/integrator/'+f'{timestamp}'+'_'+args.experiment_tag+'_'+str(int(args.nLoop/args.frames_per_sec))+'s'+"_"+str(gainCL)
    
    print('Start make env')
    os.makedirs(savedir, exist_ok=True)
    
    env.gainCL = gainCL

    print('Done change gain')
    # obj = env.reset()

    # print(type(env))
    # print(obj)

    # env.render(1)
    print("Running loop...")

    LE_PSFs = []
    SE_PSFs = []
    SRs = []
    rewards = []
    accu_reward = 0

    obs = env.reset_soft()

    # if args.anim:

    #     ATM_OPD = np.zeros((args.anim_len,) + env.atm.OPD.copy().shape)
    #     RESIDUALS = np.zeros((args.anim_len,) + env.tel.OPD.copy().shape)
    #     MIRROR_OPD = np.zeros((args.anim_len,) + env.dm.OPD.copy().shape)

    #     env.tel.computePSF(2*2, 325*2)
    #     SE_PSF = np.zeros((args.anim_len,) + env.tel.PSF_norma_zoom.copy().shape)
    for i in range(args.nLoop):
        a=time.time()
        # print(env.gainCL)
        action = env.gainCL * obs #env.integrator()
        obs, reward,strehl, done, info = env.step(i,action)  
        accu_reward+= reward

        b= time.time()
        print('Elapsed time: ' + str(b-a) +' s')
        # LE_PSF, SE_PSF = env.render(i)
        # LE_PSF, SE_PSF = env.render4plot(i)
        # env.render4plot(i)

        print('Loop '+str(i+1)+'/'+str(args.nLoop)+' Gain: '+str(gainCL)+' Turbulence: '+str(env.total[i])+' -- Residual:' +str(env.residual[i])+ '\n')
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


    # if args.anim:
    #     np.save(savedir+'/atm_opd', ATM_OPD)
    #     np.save(savedir+'/residuals', RESIDUALS)
    #     np.save(savedir+'/mirror_shape', MIRROR_OPD)
    #     np.save(savedir+'/psf', SE_PSF)


    print("Data Saved")
# %%
import torch
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn-v0_8')

exp = '/home/parker09/projects/def-lplevass/parker09/drl4papyrus/logs/finer_gains/integrator'

x = []

dirs = os.listdir(exp)



idx = np.argsort([float(x.split('.')[0][-1] + '.' + x.split('.')[1]) for x in dirs])


for i in idx:
    try:
        x = torch.load(exp+'/'+dirs[i]+'/sr2plot.pt')
        gain = float(dirs[i].split('.')[0][-1] + '.' + dirs[i].split('.')[1])
        print(gain)
        plt.plot(x, label=f'Gain:{gain:.2f}')

    except:
        continue

plt.ylabel('Strehl Ratio')
plt.legend()
plt.show()



# %%

gains = [0.1 +0.05 * (i+1) for i in range(len(dirs))]
strehl = [np.mean(torch.load(exp+'/'+dirs[i]+'/sr2plot.pt')) for i in idx]
serr = [np.std(torch.load(exp+'/'+dirs[i]+'/sr2plot.pt')) for i in idx]
# Create the error bar plot
plt.errorbar(gains, strehl, yerr=serr, fmt='o', capsize=5, capthick=2, elinewidth=1)

# Adding labels and title
plt.xlabel('Integrator Gain')
plt.ylabel('Strehl Ratio')
plt.title('Average Integrator Performance vs Gain')

# Display the plot
plt.show()
# %%
env = get_env(args)


fig, ax = plt.subplots(3,5, figsize=(20,15))

print('Starting Loop')

phase, dataset = get_phase_dataset(env, 5)

for i in range(5):

    cax1 = ax[0,i].imshow(phase[:,:,i])
    cax2 = ax[1,i].imshow(dataset['dm'][i])
    cax3 = ax[2,i].imshow(dataset['wfs'][i])

    ax[0,i].axis('off')
    ax[1,i].axis('off')
    ax[2,i].axis('off')

    ax[0,i].set_title('Random Phase', size=15)
    ax[1,i].set_title('Phase Projected onto DM', size=15)
    ax[2,i].set_title('WFS Image', size=15)

plt.tight_layout()

plt.show()
# %%

from PO4AO.conv_models_simple import Reconstructor

net = Reconstructor(1,1,11, env.xvalid, env.yvalid)

wfs_img = torch.from_numpy(dataset['wfs'][0]).float().unsqueeze(0).unsqueeze(0)

pred = net(wfs_img)

print(f'Input dims: {wfs_img.shape}, Output dims: {pred.shape}')

plt.imshow(pred[0,0,:,:].detach().numpy())
plt.show()
# %%
