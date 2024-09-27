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

checkpoint = torch.load(model_dir+'/models/tmp/finetune_CL.pt',map_location=device)

# Make sure to use the correct network before loading the state dict
reconstructor = Unet_big(env.xvalid,env.yvalid)
# Restore the regular model and optimizer state
reconstructor.load_state_dict(checkpoint['model_state_dict'])

reconstructor.to(device)

reconstructor.eval()

args.nLoop = 5000
#%%

for c_int in [0.85, 0.8, 0.75]:

    c_net = 1. - c_int

    env.atm.generateNewPhaseScreen(1333)
    env.dm.coefs = 0

    env.tel*env.dm*env.wfs

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    # savedir = '../../logs/'+args.savedir+'/integrator/'+f'{timestamp}'+'_'+args.experiment_tag+'_'+str(int(args.nLoop/args.frames_per_sec))+'s'f'_int_percent_{c_int}'
    savedir = '../../logs/'+args.savedir+'/integrator/'+args.experiment_tag+'_'+str(int(args.nLoop/args.frames_per_sec))+'s'f'_int_percent_{c_int}'

    print('Start make env')
    os.makedirs(savedir, exist_ok=True)

    print("Running loop...")

    LE_PSFs= []
    SE_PSFs = []
    SRs = []
    SR_std = []
    rewards = []
    accu_reward = 0

    obs = env.reset_soft()

    if c_net > 0:
        wfsf = torch.tensor(env.wfs.cam.frame.copy()).float().unsqueeze(1).to(device)

    for i in range(args.nLoop):
        a=time.time()

        int_action = env.gainCL * obs

        if c_net > 0:
            reshaped_input = wfsf.view(-1, 2, 24, 2, 24).permute(0, 1, 3,2, 4).contiguous().view(-1, 4, 24, 24)
            with torch.no_grad():
                action = c_int * int_action - c_net * np.sinh(reconstructor(reshaped_input).squeeze()).detach().cpu().numpy()

        else:
            action = int_action

        obs, reward,strehl, done, info = env.step(i,action)  

        if c_net>0:
            wfsf = torch.tensor(env.wfs.cam.frame.copy()).float().unsqueeze(1).to(device)

        accu_reward+= reward

        b= time.time()
        print('Elapsed time: ' + str(b-a) +' s')

        with open("CL_finetune.txt", "a") as f:  # 'a' mode appends to the file
            f.write(f"One training epoch took {b - a} seconds\n")


        print('Loop '+str(i+1)+'/'+str(args.nLoop)+' Gain: '+str(env.gainCL)+' Turbulence: '+str(env.total[i])+' -- Residual:' +str(env.residual[i])+ '\n')
        print("SR: " +str(strehl))
        if (i+1) % 100 == 0:
            sr, std = env.calculate_strehl_AVG()
            SRs.append(sr)
            SR_std.append(std)
            rewards.append(accu_reward)
            accu_reward = 0

    print(rewards)
    print("Saving Data")
    torch.save(rewards, os.path.join(savedir, "rewards2plot.pt"))
    torch.save(SRs, os.path.join(savedir, "sr2plot.pt"))
    torch.save(SR_std, os.path.join(savedir, "srstd2plot.pt"))

    print("Data Saved")

# %%
plt.style.use('ggplot')

sr1 = torch.load('/home/parker09/projects/def-lplevass/parker09/RLAO/logs/closed_loop_testing/integrator/20240927-121922_test_1s_int_percent_1.0/sr2plot.pt')
sr2 = torch.load('/home/parker09/projects/def-lplevass/parker09/RLAO/logs/closed_loop_testing/integrator/20240927-122042_test_1s_int_percent_0.75/sr2plot.pt')
sr3 = torch.load('/home/parker09/projects/def-lplevass/parker09/RLAO/logs/closed_loop_testing/integrator/20240927-123609_test_1s_int_percent_0.5/sr2plot.pt')
sr4 = torch.load('/home/parker09/projects/def-lplevass/parker09/RLAO/logs/closed_loop_testing/integrator/20240927-124914_test_1s_int_percent_0.25/sr2plot.pt')
sr5 = torch.load('/home/parker09/projects/def-lplevass/parker09/RLAO/logs/closed_loop_testing/integrator/20240927-125821_test_1s_int_percent_0.0/sr2plot.pt')
sr6 = torch.load('/home/parker09/projects/def-lplevass/parker09/RLAO/logs/closed_loop_testing/integrator/20240927-142153_test_1s_int_percent_0.85/sr2plot.pt')
sr7 = torch.load('/home/parker09/projects/def-lplevass/parker09/RLAO/logs/closed_loop_testing/integrator/20240927-144920_test_1s_int_percent_0.8/sr2plot.pt')
sr8 = torch.load('/home/parker09/projects/def-lplevass/parker09/RLAO/logs/closed_loop_testing/integrator/20240927-150439_test_1s_int_percent_0.7/sr2plot.pt')
sr9 = torch.load('/home/parker09/projects/def-lplevass/parker09/RLAO/logs/closed_loop_testing/integrator/20240927-151441_test_1s_int_percent_0.65/sr2plot.pt')




std1 = torch.load('/home/parker09/projects/def-lplevass/parker09/RLAO/logs/closed_loop_testing/integrator/20240927-121922_test_1s_int_percent_1.0/srstd2plot.pt')
std2 = torch.load('/home/parker09/projects/def-lplevass/parker09/RLAO/logs/closed_loop_testing/integrator/20240927-122042_test_1s_int_percent_0.75/srstd2plot.pt')
std3 = torch.load('/home/parker09/projects/def-lplevass/parker09/RLAO/logs/closed_loop_testing/integrator/20240927-123609_test_1s_int_percent_0.5/srstd2plot.pt')
std4 = torch.load('/home/parker09/projects/def-lplevass/parker09/RLAO/logs/closed_loop_testing/integrator/20240927-124914_test_1s_int_percent_0.25/srstd2plot.pt')
std5 = torch.load('/home/parker09/projects/def-lplevass/parker09/RLAO/logs/closed_loop_testing/integrator/20240927-125821_test_1s_int_percent_0.0/srstd2plot.pt')
std6 = torch.load('/home/parker09/projects/def-lplevass/parker09/RLAO/logs/closed_loop_testing/integrator/20240927-142153_test_1s_int_percent_0.85/srstd2plot.pt')
std7 = torch.load('/home/parker09/projects/def-lplevass/parker09/RLAO/logs/closed_loop_testing/integrator/20240927-144920_test_1s_int_percent_0.8/srstd2plot.pt')
std8 = torch.load('/home/parker09/projects/def-lplevass/parker09/RLAO/logs/closed_loop_testing/integrator/20240927-150439_test_1s_int_percent_0.7/srstd2plot.pt')
std9 = torch.load('/home/parker09/projects/def-lplevass/parker09/RLAO/logs/closed_loop_testing/integrator/20240927-151441_test_1s_int_percent_0.65/srstd2plot.pt')


plt.errorbar(np.arange(1,6), sr1, yerr=std1, fmt='o', capsize=5, label=f'Network {0}%')
plt.errorbar(np.arange(1,6), sr2, yerr=std2, fmt='o', capsize=5, label=f'Network {25}%')
# plt.errorbar(np.arange(1,6), sr3, yerr=std3, fmt='o', capsize=5,label=f'Network {50}%')
# plt.errorbar(np.arange(1,6), sr4, yerr=std4, fmt='o', capsize=5,label=f'Network {75}%')
# plt.errorbar(np.arange(1,6), sr5, yerr=std5, fmt='o', capsize=5, label=f'Network {100}%')
plt.errorbar(np.arange(1,6), sr6, yerr=std6, fmt='o', capsize=5, label=f'Network {15}%')
plt.errorbar(np.arange(1,6), sr7, yerr=std7, fmt='o', capsize=5, label=f'Network {20}%')
# plt.errorbar(np.arange(1,6), sr8, yerr=std8, fmt='o', capsize=5, label=f'Network {30}%')
# plt.errorbar(np.arange(1,6), sr9, yerr=std9, fmt='o', capsize=5, label=f'Network {35}%')



plt.legend()
plt.title('Mean Strehl over 100 frames')
plt.ylabel('SR')
plt.xlabel('Episode')
plt.show()
# %%
