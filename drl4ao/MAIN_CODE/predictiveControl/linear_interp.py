#%%
import os,sys
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ML_stuff.dataset_tools import read_yaml_file 
import time
import numpy as np
from types import SimpleNamespace
import matplotlib.pyplot as plt
from PO4AO.mbrl import get_env
from Plots.AO_plots import make_M2OPD
args = SimpleNamespace(**read_yaml_file('../Conf/papyrus_config.yaml'))

#%%

# Make the environment
env = get_env(args)

#%%

# Here we will test the effectiveness of linear interpolation as a
# predictive control method
m = 10        # Number of modes
delay = 1

M2OPD = make_M2OPD(env, n=4, m=m) # Make the M2OPD matrix

OPD2M = np.linalg.pinv(M2OPD)
xpupil, ypupil = np.where(env.tel.pupil == 1)

#%%

time_len = 100
num_modes = 10
modes = np.zeros((num_modes))
pred = np.zeros((time_len, 10))

env.tel.resetOPD()
env.atm.generateNewPhaseScreen(52843759)
for i in range(time_len):
    modes = np.vstack([modes, np.matmul(OPD2M, env.tel.OPD.copy()[xpupil, ypupil])])
    env.atm.update()
    if i == 1:
        modes = modes[1:]

    if i > 0:
        pred[i - 1] = 2 * modes[i] - modes[i - 1] 

mode = 1
plt.plot(pred[:-1,mode], label='Predicted Value')
plt.plot(modes[1:, mode], label='True Value')

plt.title(f'Mode #{mode + 1} (3 frames delay)')
plt.xlabel('Frame')
plt.ylabel('Modal Coefficient')

plt.legend()

plt.show()
# %%
import matplotlib.gridspec as gridspec
cmap = plt.get_cmap('inferno')

fig = plt.figure(figsize=(8, 8))
gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])

ax1 = plt.subplot(gs[0])
custom_colors = []
for mode in range(num_modes):
    bins = np.arange(-7e-8, 8e-8, 5e-9)
    counts, bins = np.histogram(pred[:-delay,mode] - modes[delay:, mode], bins=bins)
    # Plot the outline using plt.step()
    color = cmap(1 - (mode / (num_modes)))
    custom_colors.append(color)
    # plt.step(bins[:-1], counts, where='mid', color=color, label=f'Mode #{mode + 1}')
    # plt.vlines(bins[0], 0, counts[0], colors=color)  # Leftmost vertical bar
    # plt.vlines(bins[-2], 0, counts[-1], colors=color)
    ax1.hist(pred[:-delay,mode] - modes[delay :, mode], color=color, bins=bins, alpha=np.linspace(1, 0.4, 10)[mode], label=f'Mode #{mode + 1}')

ax1.set_title('Histogram of Residuals')

ax2 = plt.subplot(gs[1])  # Second subplot in the grid (smaller)
ax2.scatter(np.arange(1, num_modes + 1), np.std(pred[:-delay] - modes[delay:], axis=0)[:num_modes] , color=custom_colors, s=70, alpha=0.8)
ax2.set_title('Standard Deviation of Residuals per Mode')
ax2.set_xlabel('Mode Number')
# ax2.set_ylim(0, 3.3e-9)

# plt.title('Residual values from prediction')
ax1.legend()
plt.show()
# %%
