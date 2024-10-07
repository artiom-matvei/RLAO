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

# args.delay = 1

args.nLoop = 10000

for r0 in [0.13, 0.0866666667]:
    args.r0 = r0
    env = get_env(args)
    env.gainCL = 0.9


    for ws in [[10,12,11,15,20], [20,24,22,30,40]]:
        env.atm.windSpeed = ws
        env.atm.generateNewPhaseScreen(17)

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        # savedir = '../../logs/'+args.savedir+'/integrator/'+f'{timestamp}'+'_'+args.experiment_tag+'_'+str(int(args.nLoop/args.frames_per_sec))+'s'+"_"+f'r0_{r0}_ws_{ws}_gain_{env.gainCL}'
        savedir = '../../logs/can_delete/integrator/'+f'{timestamp}'+'_'+args.experiment_tag+'_'+str(int(args.nLoop/args.frames_per_sec))+'s'+"_"+f'r0_{r0}_ws_{ws}_gain_{env.gainCL}'

        print('Start make env')
        os.makedirs(savedir, exist_ok=True)


        print('Done change gain')

        print("Running loop...")

        LE_PSFs= []
        SE_PSFs = []
        SRs = []
        SR_std = []
        rewards = []
        frames_for_pwr = []
        accu_reward = 0

        obs = env.reset_soft()

        for i in range(args.nLoop):
            a=time.time()
            # print(env.gainCL)
            action = env.gainCL * obs #env.integrator()
            obs,_, reward,strehl, done, info = env.step(i,action)  

            accu_reward+= reward

            b= time.time()
            print('Elapsed time: ' + str(b-a) +' s')
            # LE_PSF, SE_PSF = env.render(i)
            # LE_PSF, SE_PSF = env.render4plot(i)
            # env.render4plot(i)


            print('Loop '+str(i+1)+'/'+str(args.nLoop)+' Gain: '+str(env.gainCL)+' Turbulence: '+str(env.total[i])+' -- Residual:' +str(env.residual[i])+ '\n')
            print("SR: " +str(strehl))
            if (i+1) % 500 == 0:
                sr, std = env.calculate_strehl_AVG()
                SRs.append(sr)
                SR_std.append(std)
                rewards.append(accu_reward)
                accu_reward = 0
                print(sr)

                


        print(f'R0: {r0}, wind speed: {ws}, Mean SR: {SRs[0]}')
        print(rewards)
        print("Saving Data")
        torch.save(rewards, os.path.join(savedir, "rewards2plot.pt"))
        torch.save(SRs, os.path.join(savedir, "sr2plot.pt"))
        torch.save(SR_std, os.path.join(savedir, "srstd2plot.pt"))



    print("Data Saved")
# %%
# import torch
# import matplotlib.pyplot as plt
# import numpy as np

# plt.style.use('seaborn-v0_8')

# exp = '/home/parker09/projects/def-lplevass/parker09/drl4papyrus/logs/finer_gains/integrator'

# x = []

# dirs = os.listdir(exp)



# idx = np.argsort([float(x.split('.')[0][-1] + '.' + x.split('.')[1]) for x in dirs])


# for i in idx:
#     try:
#         x = torch.load(exp+'/'+dirs[i]+'/sr2plot.pt')
#         gain = float(dirs[i].split('.')[0][-1] + '.' + dirs[i].split('.')[1])
#         print(gain)
#         plt.plot(x, label=f'Gain:{gain:.2f}')

#     except:
#         continue

# plt.ylabel('Strehl Ratio')
# plt.legend()
# plt.show()



# # %%

# gains = [0.1 +0.05 * (i+1) for i in range(len(dirs))]
# strehl = [np.mean(torch.load(exp+'/'+dirs[i]+'/sr2plot.pt')) for i in idx]
# serr = [np.std(torch.load(exp+'/'+dirs[i]+'/sr2plot.pt')) for i in idx]
# # Create the error bar plot
# plt.errorbar(gains, strehl, yerr=serr, fmt='o', capsize=5, capthick=2, elinewidth=1)

# # Adding labels and title
# plt.xlabel('Integrator Gain')
# plt.ylabel('Strehl Ratio')
# plt.title('Average Integrator Performance vs Gain')

# # Display the plot
# plt.show()
# # %%
# env = get_env(args)


# fig, ax = plt.subplots(3,5, figsize=(20,15))

# print('Starting Loop')

# phase, dataset = get_phase_dataset(env, 5)

# for i in range(5):

#     cax1 = ax[0,i].imshow(phase[:,:,i])
#     cax2 = ax[1,i].imshow(dataset['dm'][i])
#     cax3 = ax[2,i].imshow(dataset['wfs'][i])

#     ax[0,i].axis('off')
#     ax[1,i].axis('off')
#     ax[2,i].axis('off')

#     ax[0,i].set_title('Random Phase', size=15)
#     ax[1,i].set_title('Phase Projected onto DM', size=15)
#     ax[2,i].set_title('WFS Image', size=15)

# plt.tight_layout()

# plt.show()
# # %%

# from PO4AO.conv_models_simple import Reconstructor

# net = Reconstructor(1,1,11, env.xvalid, env.yvalid)

# wfs_img = torch.from_numpy(dataset['wfs'][0]).float().unsqueeze(0).unsqueeze(0)

# pred = net(wfs_img)

# print(f'Input dims: {wfs_img.shape}, Output dims: {pred.shape}')

# plt.imshow(pred[0,0,:,:].detach().numpy())
# plt.show()
# %%
plt.style.use('ggplot')

rl = torch.load('/home/parker09/projects/def-lplevass/parker09/RLAO/logs/papyrus_results/po4ao/20240918-165633_test_20sr0_0.0866666667_ws_[20, 24, 22, 30, 40]/sr2plot.pt')



int_09 = torch.load('/home/parker09/projects/def-lplevass/parker09/RLAO/logs/papyrus_results/integrator/20240918-180118_test_20s_r0_0.0866666667_ws_[20, 24, 22, 30, 40]_gain_0.9/sr2plot.pt')
# std_09 = torch.load('/home/parker09/projects/def-lplevass/parker09/RLAO/logs/papyrus_results/integrator/20240918-115125_test_1s_r0_0.13_ws_[10, 12, 11, 15, 20]_gain_0.9/srstd2plot.pt')
int_03 = torch.load('/home/parker09/projects/def-lplevass/parker09/RLAO/logs/papyrus_results/integrator/20240918-185105_test_20s_r0_0.0866666667_ws_[20, 24, 22, 30, 40]_gain_0.3/sr2plot.pt')
# std_03 = torch.load('/home/parker09/projects/def-lplevass/parker09/RLAO/logs/papyrus_results/integrator/20240918-120114_test_1s_r0_0.13_ws_[10, 12, 11, 15, 20]_gain_0.3/srstd2plot.pt')
# int_high = np.array([0.6272068069578972, 0.6101089753904848, 0.6466121299543616, 0.6156680442499712, 0.5975168685029444, 0.5645924791915994, 0.6135951834647645, 0.5938773426621287, 0.6281442138420543, 0.6060260610290502, 0.6427900938099987, 0.6067561657014292, 0.5962182234646175, 0.5848466731584152, 0.6166977710391496, 0.612063170573298, 0.603946121391644, 0.5721899937647568, 0.5785997720906566, 0.6240568186429183])
# int_low = np.array([0.3085460076971307, 0.29346112888697656, 0.277581243388001, 0.3003261896768564, 0.30046795740171467, 0.2882446577708787, 0.2604695290316118, 0.2750211906366203, 0.29278651507272035, 0.24276377135574356, 0.3006226846595084, 0.24657455688594143, 0.2824618668313091, 0.2689726298871343, 0.2700150266799032, 0.2266344915888839, 0.30294277338781284, 0.2899828868486627, 0.35164186520181884, 0.27791627269167446])

x = np.arange(1, len(rl)+1)

plt.plot(x, rl, label='PO4AO')

plt.plot(x,int_09, color='#348ABD',label="Integrator - 0.9 gain")
plt.plot(x, int_03, color='#FBC15E' , label="Integrator - 0.3 gain")

plt.ylim(0,0.8)
plt.axvline(2, ls='--', color='k', alpha=0.5)
plt.text(2.5, 0.7, 'Warmup', alpha=0.5)


plt.title(f'Mean Strehl of PO4AO -- r0: {0.086}, WS: {20}')
plt.legend()
plt.show()
# %%
