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

import time
import numpy as np
from PO4AO.mbrl_funcsRAZOR import get_env
from types import SimpleNamespace
import matplotlib.pyplot as plt
# SimpleNamespace takes a dict and allows the use of
# keys as attributes. ex: args['r0'] -> args.r0
args = SimpleNamespace(**read_yaml_file('Conf/razor_config_po4ao.yaml'))

#%%

env = get_env(args)

env.change_mag(4.5)


# for gainCL in args.gain_list:
for threshold in [0.01, 0.215789]:

    env.wfs.threshold_cog = threshold

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    savedir = '../../logs/'+args.savedir+'/integrator/'+f'{timestamp}'+'_'+args.experiment_tag+'_'+str(int(args.nLoop/args.frames_per_sec))+'s'+"_"+str(threshold)
    
    print('Start make env')
    os.makedirs(savedir, exist_ok=True)


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
    torch.save(rewards, os.path.join(savedir, "rewards2plot.pt"))
    torch.save(SRs, os.path.join(savedir, "sr2plot.pt"))


    # if args.anim:
    #     np.save(savedir+'/atm_opd', ATM_OPD)
    #     np.save(savedir+'/residuals', RESIDUALS)
    #     np.save(savedir+'/mirror_shape', MIRROR_OPD)
    #     np.save(savedir+'/psf', SE_PSF)


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

rl_low = torch.load('/home/parker09/projects/def-lplevass/parker09/RLAO/logs/reproduce_results/po4ao/20240915-214938_thresholds_20s_0.01/sr2plot.pt')
rl_high = torch.load('/home/parker09/projects/def-lplevass/parker09/RLAO/logs/reproduce_results/po4ao/20240915-220218_thresholds_20s_0.131/sr2plot.pt')

# int_low = torch.load('/home/parker09/projects/def-lplevass/parker09/RLAO/logs/reproduce_results/integrator/20240915-210935_thresholds_20s_0.01/sr2plot.pt')
# int_high = torch.load('/home/parker09/projects/def-lplevass/parker09/RLAO/logs/reproduce_results/integrator/20240915-212415_thresholds_20s_0.131/sr2plot.pt')

int_high = np.array([0.6272068069578972, 0.6101089753904848, 0.6466121299543616, 0.6156680442499712, 0.5975168685029444, 0.5645924791915994, 0.6135951834647645, 0.5938773426621287, 0.6281442138420543, 0.6060260610290502, 0.6427900938099987, 0.6067561657014292, 0.5962182234646175, 0.5848466731584152, 0.6166977710391496, 0.612063170573298, 0.603946121391644, 0.5721899937647568, 0.5785997720906566, 0.6240568186429183])
int_low = np.array([0.3085460076971307, 0.29346112888697656, 0.277581243388001, 0.3003261896768564, 0.30046795740171467, 0.2882446577708787, 0.2604695290316118, 0.2750211906366203, 0.29278651507272035, 0.24276377135574356, 0.3006226846595084, 0.24657455688594143, 0.2824618668313091, 0.2689726298871343, 0.2700150266799032, 0.2266344915888839, 0.30294277338781284, 0.2899828868486627, 0.35164186520181884, 0.27791627269167446])
plt.plot(rl_low, label='PO4AO - low threshold')
plt.plot(rl_high, label='PO4AO - high threshold')

x = np.arange(len(rl_low))


mean_high = np.mean(int_high)
mean_low = np.mean(int_low)

sigma_high = np.std(int_high)
sigma_low = np.std(int_low)

y_1sigma_upper_high = np.full(len(x), mean_high + sigma_high)
y_1sigma_lower_high = np.full(len(x), mean_high - sigma_high)
y_2sigma_upper_high = np.full(len(x), mean_high + 2*sigma_high)
y_2sigma_lower_high = np.full(len(x), mean_high - 2*sigma_high)

y_1sigma_upper_low = np.full(len(x), mean_low + sigma_low)
y_1sigma_lower_low = np.full(len(x), mean_low - sigma_low)
y_2sigma_upper_low = np.full(len(x), mean_low + 2*sigma_low)
y_2sigma_lower_low = np.full(len(x), mean_low - 2*sigma_low)

plt.fill_between(x, y_2sigma_lower_high, y_2sigma_upper_high, alpha=0.3)
plt.fill_between(x, y_1sigma_lower_high, y_1sigma_upper_high, alpha=0.5)


plt.fill_between(x, y_2sigma_lower_low, y_2sigma_upper_low, alpha=0.3)
plt.fill_between(x, y_1sigma_lower_low, y_1sigma_upper_low, alpha=0.5)


plt.title('Mean Strehl of PO4AO')
plt.legend()
plt.show()
# %%
