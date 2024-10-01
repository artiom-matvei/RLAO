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
from ML_stuff.dataset_tools import read_yaml_file #TorchWrapper, 
from ML_stuff.models import Unet_big
from Plots.gifTools import create_gif

import time
import numpy as np

from types import SimpleNamespace
import matplotlib.pyplot as plt
# SimpleNamespace takes a dict and allows the use of
# keys as attributes. ex: args['r0'] -> args.r0
#For razor sim
# from PO4AO.mbrl_funcsRAZOR import get_env
# args = SimpleNamespace(**read_yaml_file('Conf/razor_config_po4ao.yaml'))

#For papyrus sim
from PO4AO.mbrl import get_env
args = SimpleNamespace(**read_yaml_file('Conf/papyrus_config.yaml'))
#%%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_dir = '/home/parker09/projects/def-lplevass/parker09/RLAO/drl4ao/MAIN_CODE/wf_recon'

env = get_env(args)

checkpoint = torch.load(model_dir+'/models/useable/finetune_CL.pt',map_location=device)

# Make sure to use the correct network before loading the state dict
reconstructor = Unet_big(env.xvalid,env.yvalid)
# Restore the regular model and optimizer state
reconstructor.load_state_dict(checkpoint['model_state_dict'])

reconstructor.to(device)

reconstructor.eval()

args.nLoop = 1000

env.tel.resetOPD()
env.tel.computePSF(4)

psf_model_max = env.tel.PSF.max()

#%%



for c_int in [1, 0.6, 0.]:# [1, 0.85, 0.8, 0.75]:

    c_net = 1. - c_int


    timestamp = time.strftime("%Y%m%d-%H%M%S")
    # savedir = '../../logs/'+args.savedir+'/integrator/'+f'{timestamp}'+'_'+args.experiment_tag+'_'+str(int(args.nLoop/args.frames_per_sec))+'s'f'_int_percent_{c_int}'
    savedir = '../../logs/'+args.savedir+'/integrator/'+args.experiment_tag+'_'+str(int(args.nLoop/args.frames_per_sec))+'s'f'_int_percent_{c_int}'

    print('Start make env')
    os.makedirs(savedir, exist_ok=True)

    for j in range(10):
        print("Running loop...")

        env.atm.generateNewPhaseScreen(9323 * j)
        env.dm.coefs = 0

        env.tel*env.dm*env.wfs


        LE_PSFs= []
        SE_PSFs = []
        SRs = []
        SR_std = []
        rewards = []
        accu_reward = 0

        LE_SR = []

        use_net = True
        reset_counter = 0

        obs = env.reset_soft()

        if c_net > 0:
            wfsf = torch.tensor(env.wfs.cam.frame.copy()).float().unsqueeze(1).to(device)

        for i in range(args.nLoop):
            a=time.time()

            reset_counter += 1

            int_action = env.gainCL * obs

            if (c_net > 0)&(use_net):
                reshaped_input = wfsf.view(-1, 2, 24, 2, 24).permute(0, 1, 3,2, 4).contiguous().view(-1, 4, 24, 24)
                with torch.no_grad():
                    tensor_output = reconstructor(reshaped_input).squeeze().detach().cpu()
                    numpy_output = np.sinh(tensor_output.numpy())  # Now convert to NumPy
                    action = c_int * int_action - c_net * numpy_output

            else:
                action = int_action

            obs,_, reward,strehl, done, info = env.step(i,action)  

            if c_net>0:
                wfsf = torch.tensor(env.wfs.cam.frame.copy()).float().unsqueeze(1).to(device)

            accu_reward+= reward

            b= time.time()
            print('Elapsed time: ' + str(b-a) +' s')

            env.tel.computePSF(4)
            SE_PSFs.append(env.tel.PSF)

            LE_SR.append(np.max(np.mean(SE_PSFs, axis=0))/psf_model_max)


            print('Loop '+str(i+1)+'/'+str(args.nLoop)+' Gain: '+str(env.gainCL)+' Turbulence: '+str(env.total[i])+' -- Residual:' +str(env.residual[i])+ '\n')
            print("SR: " +str(strehl))
            if (i+1) % 100 == 0:
                sr, std = env.calculate_strehl_AVG()
                SRs.append(sr)
                SR_std.append(std)
                rewards.append(accu_reward)
                accu_reward = 0

                use_net = True
                reset_counter = 0

        print(rewards)
        print("Saving Data")
        torch.save(rewards, os.path.join(savedir, "rewards2plot.pt"))
        torch.save(SRs, os.path.join(savedir, "sr2plot.pt"))
        torch.save(SR_std, os.path.join(savedir, "srstd2plot.pt"))
        torch.save(LE_SR, os.path.join(savedir, f"LE_SR_{j}.pt"))

        print("Data Saved")

# %%
plt.style.use('ggplot')

x = [0.85, 0.8,  0.75, 0.7,  0.65, 0.6,  0.55, 0.5,  0.45, 0.4,  0.35,
 0.3,  0.25, 0.2,  0.15, 0.1,  0.05]

lsr1 = torch.load(f'/home/parker09/projects/def-lplevass/parker09/RLAO/logs/edge_correction_leak/integrator/test_10s_int_percent_1/sr2plot.pt')
slsr1 = torch.load('/home/parker09/projects/def-lplevass/parker09/RLAO/logs/edge_correction_leak/integrator/test_10s_int_percent_1/srstd2plot.pt')
plt.errorbar(np.arange(1,51), lsr1, yerr=slsr1, fmt='o', capsize=5, label=f'Network {0}%')


for i in x[5:8]:
    slsr2 = torch.load(f'/home/parker09/projects/def-lplevass/parker09/RLAO/logs/edge_correction_leak/integrator/test_10s_int_percent_{i}/srstd2plot.pt')
    
    lsr2 = torch.load(f'/home/parker09/projects/def-lplevass/parker09/RLAO/logs/edge_correction_leak/integrator/test_10s_int_percent_{i}/sr2plot.pt')
    plt.errorbar(np.arange(1,51), lsr2, yerr=slsr2, fmt='o', capsize=5, label=f'Network {(1 - i)*100:.0f}%')

plt.legend()
plt.title('Mean Strehl over 100 frames')
plt.ylabel('SR')
plt.xlabel('Episode')
plt.xlim(30,40)
plt.show()
# %%
plt.style.use('ggplot')

x = [0.0,0.6,1]

for i in x:
    slsr2 = torch.load(f'/home/parker09/projects/def-lplevass/parker09/RLAO/logs/LE_compare/integrator/test_2s_int_percent_{i}/LE_SR.pt')
    print(slsr2)
    plt.plot(slsr2, label=f'Network {(1 - i)*100:.0f}%')

plt.legend()
plt.title('Long Exposure SR')
plt.ylabel('SR')
plt.xlabel('Frame Number')
# plt.xlim(30,40)
plt.show()
# %%
