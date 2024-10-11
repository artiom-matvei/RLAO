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
# from Plots.AO_plots import make_M2OPD
args = SimpleNamespace(**read_yaml_file('../../Conf/papyrus_config.yaml'))

#%%
args.modulation = 3
# Make the environment
env = get_env(args)

#%%

# Here we will test the capabilities of the EOFs on objective measurements of the wavefront

# First we need to make a dataset D, each column of this matrix is a vector h, which is organized as follows:
# h(t) = [w_0(t), w_1(t), ..., w_{m-1}(t), w_0(t - dt),..., w_{m-1}(t - dt), ..., w_{m-1}(t - (n-1)dt)]
# where w_i(t) is the i-th mode of the wavefront at time t, and dt is the time step between each measurement

# Since we will start by just correcting TT (m = 2), we will pick n = 10 to try out

m = 10        # Number of modes
n = 10       # Number of time steps
l = 1000     # Number of samples
delay = 1   # Delay measurement and prediction

D = np.zeros((m*n, l))        # Initialize the dataset

P = np.zeros((m, l))          # Initialize prediction matrix

#%%

print('Starting dataset generation')

for i in range(l):
    env.tel.resetOPD()
    env.atm.generateNewPhaseScreen(np.random.randint(0, 2**32 - 2))
    for j in range(n):
        env.atm.update()
        env.tel*env.wfs

        coefs = np.matmul(env.modal_CM, env.wfs.signal)

        for k in range(m):
            D[j + k*n, i] = coefs[k]

    for k in range(delay):
        env.atm.update()
        env.tel*env.wfs
    
    coefs = np.matmul(env.modal_CM, env.wfs.signal)

    for k in range(m):
        P[k, i] = coefs[k]

    if (i+1) % 100 == 0:
        print(f'{i+1} samples generated')


# The optimal filter is given by F = ((D^T)^+ P^T)^T
# We will use the SVD to compute the pseudo-inverse of D^T

# U, s, Vt = np.linalg.svd(D.T, full_matrices=False)

# threshold = 0.01 * s[0]

# # Apply threshold by setting small singular values to zero
# s_thresholded = np.where(s > threshold, s, 0)

# # Invert non-zero singular values for pseudoinverse
# s_pinv = np.where(s_thresholded > 0, 1 / s_thresholded, 0)

# # Reconstruct the pseudoinverse using the thresholded singular values
# Sigma_pinv = np.diag(s_pinv)

# # Compute pseudoinverse
# A_pinv = Vt.T @ Sigma_pinv @ U.T

print('Starting EOF computation')

F = np.zeros((m, m*n))

for i in range(m):
    F[i] = ((np.linalg.pinv(D.T) @ P[i].T).T)




# %%
F = np.load('/Users/parkerlevesque/School/Research/AO/RLAO/drl4ao/MAIN_CODE/predictiveControl/saved_filters/F_5k_10h_10m_3d.npy')
delay = 3

time_len = 100
num_modes = 10
modes = np.zeros((num_modes))
pred = np.zeros((time_len - n + 1, 10))

env.tel.resetOPD()
env.atm.generateNewPhaseScreen(52843759)
for i in range(time_len):
    modes = np.vstack([modes, np.matmul(OPD2M, env.tel.OPD.copy()[xpupil, ypupil])])
    env.atm.update()
    if i == 1:
        modes = modes[1:]

    if (i+1) >= n:
        obs = np.zeros(n * num_modes)
        for k in range(num_modes):
            obs[k * n: k* n + n] = modes[-n:, k]

        pred[i - (n-1)] = F@ obs

mode = 1
plt.plot(pred[:-delay,mode], label='Predicted Value')
plt.plot(modes[n + delay - 1:, mode], label='True Value')

plt.title(f'Mode #{mode + 1} (3 frames delay)')
plt.xlabel('Frame')
plt.ylabel('Modal Coefficient')

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
    counts, bins = np.histogram(pred[:-delay,mode] - modes[n + delay - 1:, mode], bins=bins)
    # Plot the outline using plt.step()
    color = cmap(1 - (mode / (num_modes)))
    custom_colors.append(color)
    # plt.step(bins[:-1], counts, where='mid', color=color, label=f'Mode #{mode + 1}')
    # plt.vlines(bins[0], 0, counts[0], colors=color)  # Leftmost vertical bar
    # plt.vlines(bins[-2], 0, counts[-1], colors=color)
    ax1.hist(pred[:-delay,mode] - modes[n + delay - 1:, mode], color=color, bins=bins, alpha=np.linspace(1, 0.4, 10)[mode], label=f'Mode #{mode + 1}')

ax1.set_title('Histogram of Residuals')

ax2 = plt.subplot(gs[1])  # Second subplot in the grid (smaller)
ax2.scatter(np.arange(1, num_modes + 1), np.std(pred[:-delay] - modes[n + delay - 1:], axis=0)[:num_modes] , color=custom_colors, s=70, alpha=0.8)
ax2.set_title('Standard Deviation of Residuals per Mode')
ax2.set_xlabel('Mode Number')
ax2.set_ylim(0, 3.3e-9)

# plt.title('Residual values from prediction')
ax1.legend()
plt.show()
# %%
