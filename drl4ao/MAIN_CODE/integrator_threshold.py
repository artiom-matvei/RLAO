"""
OOPAO module for the integrator
@author: Raissa Camelo (LAM) git: @srtacamelo
"""

#%%
import os,sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

# from parser_Configurations import Config, ConfigAction
# from OOPAOEnv.OOPAOEnvRazor import OOPAO
from PO4AO.util_simple import read_yaml_file #TorchWrapper, 
from Plots.plots import save_plots
# import matplotlib.pyplot as plt
# import argparse
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

# %%

mags = np.arange(3, 6, 0.5)

for mag in mags:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    savedir = '../../logs/'+args.savedir+'/integrator/'+f'{timestamp}'+'_'+args.experiment_tag+'_'+f'magnitude_{mag}'+f'_gain_{env.gainCL}'
    
    print('Start make env')
    os.makedirs(savedir, exist_ok=True)

    # SET TO DEFAULT VALUE
    env.wfs.threshold_cog = 0.01

    #UPDATE MAGNITUDE
    env.change_mag(mag)
    np.save(savedir+'/raw_wfs', env.wfs.cam.frame.copy())

    # COMPUTE OPTIMAL THRESHOLD
    wfs_max = np.max(env.wfs.cam.frame.copy())
    wfs_min = np.min(env.wfs.cam.frame.copy())

    optimal_thresh = -wfs_min / wfs_max

    thresh_mesh = np.linspace(0.01, optimal_thresh, 6)

    # PLOT DEFAULT VS OPTIMAL ON HIST
    # plt.figure(mag)
    # frame = env.wfs.cam.frame.copy()
    # bins = np.arange(np.min(frame), np.max(frame)+ 3)
    # plt.hist(frame.flatten(), bins=bins)

    # plt.axvline(0.01 * wfs_max, color='r', label='default threshold')
    # plt.axvline(- wfs_min, color='orange', label = 'optimal threshold')

    # plt.yscale('log')
    # plt.legend()
    # plt.title('WFS Signal with Centroid Thresholds')
    # plt.savefig(savedir+'/wfs_hist.png')

    print('Done change magnitude')

    print("Running loop...")

    LE_PSFs = []
    SE_PSFs = []
    SRs = []
    rewards = []
    accu_reward = 0

    obs = env.reset_soft()

    for thresh in thresh_mesh:
        env.wfs.threshold_cog = thresh

        env.plot_wfs()
        np.save(savedir+f'/slope_map_thresh_{thresh:.2f}', env.wfs.signal_2D.copy())

        for i in range(args.nLoop):
            a=time.time()
            # print(env.gainCL)
            action = env.gainCL * obs #env.integrator()
            obs, reward,strehl, done, info = env.step(i,action)  
            accu_reward+= reward

            b= time.time()
            print('Elapsed time: ' + str(b-a) +' s')

            # print('Loop '+str(i+1)+'/'+str(args.nLoop)+' Gain: '+str(env.gainCL)+' Turbulence: '+str(env.total[i])+' -- Residual:' +str(env.residual[i])+ '\n')
            # print("SR: " +str(strehl))
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
# import torch
# import matplotlib.pyplot as plt
# import numpy as np

# plt.style.use('seaborn-v0_8')

# exp = '/home/parker09/projects/def-lplevass/parker09/drl4papyrus/logs/threshold_refSlopes/integrator'

# x = []

# dirs = os.listdir(exp)



# idx = np.argsort([float(x[-3:]) for x in dirs])


# for i in idx:
#     try:
#         x = torch.load(exp+'/'+dirs[i]+'/sr2plot.pt')
#         magnitude = float(dirs[i][-3:])

#         plt.plot(x, label=f'Source mag:{magnitude:.1f}')

#     except:
#         continue

# plt.ylabel('Strehl Ratio')
# plt.legend()
# plt.show()



# # %%

# before = np.array([torch.load(exp+'/'+dirs[i]+'/sr2plot.pt')[:10] for i in idx])
# after  = np.array([torch.load(exp+'/'+dirs[i]+'/sr2plot.pt')[10:] for i in idx])



# mags = np.arange(3,6,0.5)

# # Create the error bar plot
# plt.plot(mags,np.mean(before, axis=1), 'ro', label='default threshold')
# plt.plot(mags,np.mean(after, axis=1), 'ko', label='optimal threshold')

# # Adding labels and title
# plt.xlabel('Source Magnitude')
# plt.ylabel('Strehl Ratio')
# plt.title('Average Integrator Performance vs Source Magnitude')

# plt.legend()
# # Display the plot
# plt.show()
# # %%


# #UPDATE MAGNITUDE
# env.change_mag(4)

# # COMPUTE OPTIMAL THRESHOLD
# wfs_max = np.max(env.wfs.cam.frame.copy())
# wfs_min = np.min(env.wfs.cam.frame.copy())

# optimal_thresh = -wfs_min / wfs_max

# env.wfs.threshold_cog = 0.01#optimal_thresh

# env.plot_wfs(threshold=optimal_thresh)

# env.atm.update()
# env.tel*env.wfs

# env.tel.resetOPD()
# env.tel*env.wfs

# plt.imshow(env.wfs.signal_2D.copy())
# # %%
# np.save('/home/parker09/projects/def-lplevass/parker09/drl4papyrus/logs/wfs_images/4mag_raw',env.wfs.cam.frame.copy() )
# np.save('/home/parker09/projects/def-lplevass/parker09/drl4papyrus/logs/wfs_images/4mag_slopes',env.wfs.signal_2D.copy() )
# # %%

# env.tel.resetOPD()
# env.tel*env.wfs

# before = env.wfs.cam.frame.copy()


# plt.imshow(env.tel.OPD)
# plt.show()
# plt.imshow(before)


# env.atm.update()
# env.tel*env.wfs

# after = env.wfs.cam.frame.copy()

# plt.show()
# plt.imshow(env.tel.OPD)
# plt.show()
# plt.imshow(env.wfs.cam.frame)
# plt.show()
# plt.imshow(after - before)
# plt.show()
# %%
data = torch.load('/home/parker09/projects/def-lplevass/parker09/drl4papyrus/logs/many_threshold/integrator/20240907-172744_gains_magnitude_4.5_gain_0.2/sr2plot.pt')

arr = np.array(data).reshape(6,10)
srs = np.mean(arr, axis=1)

wfs_raw = np.load('/home/parker09/projects/def-lplevass/parker09/drl4papyrus/logs/many_threshold/integrator/20240907-172744_gains_magnitude_4.5_gain_0.2/raw_wfs.npy')


wfs_max = np.max(wfs_raw)
wfs_min = np.min(wfs_raw)
ot = -wfs_min/wfs_max

threshs = np.linspace(0.01, ot, 6)

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 9))  # 2 rows, 1 column

# First plot (top)
ax1.plot(threshs*wfs_max, np.mean(arr, axis=1))
ax1.set_title('Average Strehl')

# Second plot (middle)
ax2.hist(wfs_raw.flatten(), bins=np.arange(wfs_min, wfs_max))
for i in threshs:
    ax2.axvline(i*wfs_max, c='r')
ax2.set_yscale('log')
ax2.set_title('Threshold location')

ax1.set_xlim(threshs[0]*wfs_max - 1, threshs[-1]*wfs_max + 1)
ax2.set_xlim(threshs[0]*wfs_max - 1, threshs[-1]*wfs_max + 1)

# Third Plot (bottom)
ax3.hist(wfs_raw.flatten(), bins=np.arange(wfs_min, wfs_max + 2))
for i in threshs:
    ax3.axvline(i*wfs_max, c='r')
ax3.set_yscale('log')
ax3.set_title('Zoomed Out Version')


# Adjust layout for better spacing
plt.tight_layout()

# Display the plot
plt.show()
# %%
