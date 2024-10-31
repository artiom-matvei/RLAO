#%%
import os,sys
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from ML_stuff.dataset_tools import read_yaml_file 
import time
import numpy as np
from types import SimpleNamespace
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from PO4AO.mbrl import get_env
from Plots.AO_plots import make_M2OPD
args = SimpleNamespace(**read_yaml_file('../../Conf/papyrus_config.yaml'))

#%%

# Make the environment
env = get_env(args)

#%%

# Here we will test the effectiveness of linear interpolation as a
# predictive control method
m = 300        # Number of modes
n = 2         # Number of time steps needed for prediction
delay = 1

M2OPD = np.load('/Users/parkerlevesque/School/Research/AO/RLAO/drl4ao/MAIN_CODE/predictiveControl/saved_filters/M2OPD_300modes.npy')

OPD2M = np.linalg.pinv(M2OPD)
xpupil, ypupil = np.where(env.tel.pupil == 1)



#%%

time_len = 1000
num_modes = 300
modes = np.zeros((num_modes))
pred = np.zeros((time_len - n + 1, num_modes))

env.tel.resetOPD()
env.atm.generateNewPhaseScreen(52843759)
for i in range(time_len):
    modes = np.vstack([modes, np.matmul(OPD2M, env.tel.OPD.copy()[xpupil, ypupil])])
    env.atm.update()
    if i == 1:
        modes = modes[1:]

    if (i+1) >= n:
        pred[i - (n-1)] = (delay + 1) * modes[i] - delay * modes[i - 1] 

    if i % 10 == 0:
        print(f'Completed {i} iterations')

mode = 1
plt.plot(pred[:-delay,mode], label='Predicted Value')
plt.plot(modes[n + delay - 1:, mode], label='True Value')

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
    bins = np.arange(-7e-9, 8e-9, 5e-10)
    # counts, bins = np.histogram(pred[:-delay,mode] - modes[n + delay - 1:, mode], bins=bins)
    # # Plot the outline using plt.step()
    color = cmap(1 - (mode / (num_modes)))
    custom_colors.append(color)
    # plt.step(bins[:-1], counts, where='mid', color=color, label=f'Mode #{mode + 1}')
    # plt.vlines(bins[0], 0, counts[0], colors=color)  # Leftmost vertical bar
    # plt.vlines(bins[-2], 0, counts[-1], colors=color)
    ax1.hist(pred[:-delay,mode] - modes[n + delay - 1 :, mode], color=color, bins=bins, alpha=np.linspace(1, 0.4, 10)[mode], label=f'Mode #{mode + 1}')

ax1.set_title('Histogram of Residuals')

ax2 = plt.subplot(gs[1])  # Second subplot in the grid (smaller)
ax2.scatter(np.arange(1, num_modes + 1), np.std(pred[:-delay] - modes[n + delay - 1 :], axis=0)[:num_modes] , color=custom_colors, s=70, alpha=0.8)
ax2.set_title('Standard Deviation of Residuals per Mode')
ax2.set_xlabel('Mode Number')
ax2.set_ylim(1e-10, 5e-8)
ax2.set_yscale('log')
# plt.title('Residual values from prediction')
ax1.legend()
plt.show()
# %%


dt = 1/500

ir = np.zeros(300)

for mode in range(300):


    derivative = np.diff(modes[:, mode]) / dt

    # Step 2: Identify zero-crossings (changes in sign)
    zero_crossings = np.where(np.diff(np.sign(derivative)) != 0)[0]

    # Step 3: Count the number of inversions
    num_inversions = len(zero_crossings)

    # Step 4: Calculate the rate of inversion
    total_time = len(modes[:, mode]) * dt
    rate_of_inversion = num_inversions / total_time

    ir[mode] = rate_of_inversion

plt.plot(ir, ls='', marker='o', ms=5)
plt.xlabel('Mode')
plt.ylabel('Inversion Rate')

plt.title('Rate of Inversion per Mode')

from scipy.optimize import curve_fit

def power_law(x, a, b):
    return a * x**b

# Fit the data to the power law function
params, covariance = curve_fit(power_law, np.arange(0, 300), ir)

# Extract fitted parameters
a_fit, b_fit = params

# Generate y values based on the fitted function for plotting
x_fit = np.arange(0, 300)
y_fit = power_law(x_fit, a_fit, b_fit)

import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(8, 8))
gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])

ax1 = plt.subplot(gs[0])
custom_colors = []

ax1.scatter(np.arange(0, 300), ir, label='Data Points', color='blue')
ax1.plot(x_fit, y_fit, label=f'Fit: y = {a_fit:.2f} * x^{b_fit:.2f}', color='red')

ax1.set_title('Inversion Rate vs Mode')

ax2 = plt.subplot(gs[1])  # Second subplot in the grid (smaller)
ax2.plot(np.arange(1, 300+1), np.arange(1, 300+1)**(-b_fit))
ax2.set_title('Alpha Gain vs Mode')
ax2.set_xlabel('Mode Number')
# plt.title('Residual values from prediction')
ax1.legend()
plt.show()