"""
@author: Parker Levesque
"""

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
from linear_extrap.LEPredict import LEPredictiveModel
args = SimpleNamespace(**read_yaml_file('../Conf/papyrus_config.yaml'))


#%%
args.delay = 2
args.modulation = 3
args.nLoop = 1000

env = get_env(args)
env.gainCL = 0.9

env.tel.resetOPD()
env.tel.computePSF(4)

psf_model_max = env.tel.PSF.max()


LEPred = LEPredictiveModel(modal=True, num_modes=50)

env.atm.generateNewPhaseScreen(17)

#%%

timestamp = time.strftime("%Y%m%d-%H%M%S")
# savedir = '../../logs/'+args.savedir+'/integrator/'+f'{timestamp}'+'_'+args.experiment_tag+'_'+str(int(args.nLoop/args.frames_per_sec))+'s'+"_"+f'r0_{r0}_ws_{ws}_gain_{env.gainCL}'
savedir = '../../logs/can_delete/integrator/'+f'{timestamp}'+'_'+args.experiment_tag+'_'+str(int(args.nLoop/args.frames_per_sec))+'s'+"_"+f'gain_{env.gainCL}'

# print('Start make env')
os.makedirs(savedir, exist_ok=True)

# print("Running loop...")
pts = 5
sr_arr = np.zeros((pts, args.nLoop // 50))
le_arr = np.zeros((pts, args.nLoop))

for j, gamma in enumerate(np.linspace(0, 0.4, pts)):
    env.tel.resetOPD()
    env.dm.coefs = 0

    env.atm.update()

    env.tel*env.dm*env.wfs

    # LE cmd = DMC(t-1) + 2R(t) - R(t-1)

    env.atm.generateNewPhaseScreen(175)

    LE_PSFs= []
    SE_PSFs = []
    LE_SR = []
    SRs = []
    SR_std = []
    rewards = []
    frames_for_pwr = []
    accu_reward = 0

    states = []
    pred_states = []

    use_integrator = False

    time_len = args.nLoop
    delay = args.delay

    n = 5
    m = 50 #357
    obs4pred = np.zeros(n*m)
    buffer = np.zeros((n,357))
    pred = np.zeros((time_len - n + 1, m))

    M2OPD = np.load('./saved_filters/M2OPD_300modes.npy') # Make the M2OPD matrix

    OPD2M = np.linalg.pinv(M2OPD)
    xpupil, ypupil = np.where(env.tel.pupil == 1)

    C2M = np.linalg.pinv(env.M2C_CL.copy())




    obs = env.reset_soft()

    obs_buffer = np.zeros((2, m))
    obs_buffer[0] = C2M@env.img_to_vec(obs).numpy() # obs


    for i in range(args.nLoop):
        a=time.time()


        if i < 20:
            action = env.gainCL * obs 

            prev_action = C2M@env.img_to_vec(action).numpy()


        elif use_integrator:
            action = env.gainCL * obs 

            prev_action = C2M@env.img_to_vec(action).numpy()


        else:
            action = env.gainCL * obs - gamma * env.vec_to_img(env.M2C_CL@LEPred.predict_next_command(obs_buffer, prev_action))

            prev_action = C2M@env.img_to_vec(action).numpy()


        obs,_, reward,strehl, done, info = env.step(i,torch.tensor(action)) 

        
        env.tel.computePSF(4)
        SE_PSFs.append(env.tel.PSF[175:-175, 175:-175])
        LE_PSFs.append(np.mean(SE_PSFs, axis=0))

        LE_SR.append(np.max(np.mean(SE_PSFs, axis=0))/psf_model_max)

        


        obs_buffer = np.roll(obs_buffer,1,axis=0)
        obs_buffer[0] = C2M@env.img_to_vec(obs).numpy() # obs


        DMS = env.dm.coefs.copy() * 1e6

        R = env.img_to_vec(obs).numpy()

        pol_now = DMS + R

        states.append(pol_now)

        buffer = np.roll(buffer,1,axis=0)
        buffer[0] = pol_now

        # linear extrapolation
        if i > 0:
            pred_state = 2 * buffer[0] - buffer[1]
            pred_states.append(pred_state)


        # for EOF

        # for k in range(m):
        #     # in the k-th n block, we store the k-th mode's last n measurements
        #     # going from oldest to newest
        #     obs4pred[k * n: k* n + n] = buffer[:, k]

        accu_reward+= reward

        b= time.time()
        print('Elapsed time: ' + str(b-a) +' s')

        print('Loop '+str(i+1)+'/'+str(args.nLoop)+' Gain: '+str(env.gainCL)+' Turbulence: '+str(env.total[i])+' -- Residual:' +str(env.residual[i])+ '\n')
        print("SR: " +str(strehl))
        if (i+1) % 50 == 0:
            sr, std = env.calculate_strehl_AVG()
            SRs.append(sr)
            SR_std.append(std)
            rewards.append(accu_reward)
            accu_reward = 0
            # print(sr)

            


    # print(f'Mean SR: {SRs[0]}')
    # print(rewards)
    # print("Saving Data")
    torch.save(rewards, os.path.join(savedir, "rewards2plot.pt"))
    torch.save(SRs, os.path.join(savedir, "sr2plot.pt"))
    torch.save(SR_std, os.path.join(savedir, "srstd2plot.pt"))

    sr_arr[j] = SRs
    le_arr[j] = LE_SR


# print("Data Saved")

#%% 
def moving_average(arr, n):
    return np.convolve(arr, np.ones(n)/n, mode='valid')

n = 5

pred_states = np.array(pred_states)
states = np.array(states)

residuals = pred_states[:-1] - states[2:]
no_res = states[2:] - states[1:-1]


plt.plot(moving_average(np.var(residuals, axis=1),n), label='Linear Extrapolation')
plt.plot(moving_average(np.var(no_res, axis=1),n), label='No prediction')
plt.legend()
plt.yscale('log')
plt.xlabel('Frame number')
plt.ylabel('RMSE')
plt.title('Smoothed RMSE of Predicted State vs Ground Truth')

#%%
states = np.array(states)

mode = 5
plt.plot(pred[:-delay,mode], label='Predicted Value')
plt.plot(states[n + delay - 1:, mode], label='True Value')

plt.title(f'Mode #{mode + 1} (1 frames delay)')
plt.xlabel('Frame')
plt.ylabel('Modal Coefficient')

plt.legend()

plt.show()

states = np.array(states)
import matplotlib.gridspec as gridspec
plt.style.use('ggplot')
cmap = plt.get_cmap('inferno')
num_modes = 10
fig = plt.figure(figsize=(8, 8))
gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])

ax1 = plt.subplot(gs[0])
custom_colors = []
for mode in range(num_modes):
    bins = np.arange(-0.04, 0.04 + 0.005, 0.005)
    # counts, bins = np.histogram(pred[:-delay,mode] - modes[n + delay - 1:, mode], bins=bins)
    # # Plot the outline using plt.step()
    color = cmap(1 - (mode / (num_modes)))
    custom_colors.append(color)
    # plt.step(bins[:-1], counts, where='mid', color=color, label=f'Mode #{mode + 1}')
    # plt.vlines(bins[0], 0, counts[0], colors=color)  # Leftmost vertical bar
    # plt.vlines(bins[-2], 0, counts[-1], colors=color)
    ax1.hist(pred[:-delay,mode] - states[n + delay - 1 :, mode], color=color, bins=bins, alpha=np.linspace(1, 0.4, 10)[mode], label=f'Mode #{mode + 1}')
    # ax1.set_yscale('log')

ax1.set_title('Histogram of Residuals')

ax2 = plt.subplot(gs[1])  # Second subplot in the grid (smaller)
ax2.scatter(np.arange(1, num_modes + 1), np.std(pred[:-delay] - states[n + delay - 1 :, :num_modes], axis=0)[:num_modes] , color=custom_colors, s=70, alpha=0.8)
ax2.set_title('Standard Deviation of Residuals per Mode')
ax2.set_xlabel('Mode Number')
ax2.set_ylim(1e-10, 0.5)
ax2.set_yscale('log')
# plt.title('Residual values from prediction')
ax1.legend()
plt.show()

print(SRs)

# %%
