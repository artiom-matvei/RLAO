"""
OOPAO module for the integrator
@author: Raissa Camelo (LAM) git: @srtacamelo
"""

#%%
import os,sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from ML_stuff.dataset_tools import read_yaml_file, find_main_directory
from Plots.plots import save_plots
# import matplotlib.pyplot as plt
# import argparse
import time
import numpy as np
import torch
from PO4AO.mbrl_funcsRAZOR import get_env
from types import SimpleNamespace
import matplotlib.pyplot as plt
from cycler import cycler

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

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 9))

data = torch.load(find_main_directory(__file__) + '/logs/many_threshold/integrator/20240907-172744_gains_magnitude_4.5_gain_0.2/sr2plot.pt')

arr = np.array(data).reshape(6,10)
srs = np.mean(arr, axis=1)

wfs_raw = np.load(find_main_directory(__file__) + '/logs/many_threshold/integrator/20240907-172744_gains_magnitude_4.5_gain_0.2/raw_wfs.npy')


wfs_max = np.max(wfs_raw)
wfs_min = np.min(wfs_raw)
ot = -wfs_min/wfs_max

threshs = np.linspace(0.01, ot, 6)

  # 2 rows, 1 column

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
##### MAKING THE BIG POT #####

# Custom color scheme inspired by 'inferno'
custom_colors = ['#440154', '#482878', '#F0701D', '#FF4500', '#FDE724']

# Set the color cycle globally for all plots

plt.style.use('seaborn-v0_8')
plt.rc('axes', prop_cycle=cycler('color', custom_colors))

exp = find_main_directory(__file__) + '/logs/many_threshold_plus/many_threshold_plus/integrator'

x = []

dirs = os.listdir(exp)

idx = np.argsort([float(x[-12:-9]) for x in dirs])

fig, ax = plt.subplots(3, len(dirs), figsize=(4*len(dirs), 9))
for j, i in enumerate(idx):
    data = torch.load(exp+'/'+dirs[i]+'/sr2plot.pt')

    arr = np.array(data).reshape(8,10)
    srs = np.mean(arr, axis=1)
    wfs_raw = np.load(exp+'/'+dirs[i]+'/raw_wfs.npy')
    magnitude = float(dirs[i][-12:-9])


    wfs_max = np.max(wfs_raw)
    wfs_min = np.min(wfs_raw)
    ot = -wfs_min/wfs_max

    threshs = np.linspace(0.01, ot, 6)

    step_size = threshs[1] - threshs[0]

    last_value = threshs[-1]

    thresh_mesh = np.append(threshs, [last_value + step_size, last_value + 2 * step_size])

      # 2 rows, 1 column

    # First plot (top)
    ax[0,j].plot(thresh_mesh*wfs_max, np.mean(arr, axis=1), c=custom_colors[1])
    ax[0,j].set_title(f'Mean Strehl -- Mag {magnitude}')

    # Second plot (middle)
    ax[1,j].hist(wfs_raw.flatten(), bins=np.arange(wfs_min, wfs_max), color=custom_colors[1])
    for i in thresh_mesh:
        ax[1,j].axvline(i*wfs_max, c=custom_colors[3])
    ax[1,j].set_yscale('log')
    ax[1,j].set_title('Threshold location')

    ax[0,j].set_xlim(thresh_mesh[0]*wfs_max - 1, thresh_mesh[-1]*wfs_max + 1)
    ax[1,j].set_xlim(thresh_mesh[0]*wfs_max - 1, thresh_mesh[-1]*wfs_max + 1)

    # Third Plot (bottom)
    ax[2,j].hist(wfs_raw.flatten(), bins=np.arange(wfs_min, wfs_max + 2), color=custom_colors[1])
    for i in thresh_mesh:
        ax[2,j].axvline(i*wfs_max, c=custom_colors[3])
    ax[2,j].set_yscale('log')
    ax[2,j].set_title('Zoomed Out Version')


plt.ylabel('Strehl Ratio')
# plt.legend()
# plt.savefig('../../logs/'+args.savedir+'/integrator/result.png')
plt.show()
# %%
