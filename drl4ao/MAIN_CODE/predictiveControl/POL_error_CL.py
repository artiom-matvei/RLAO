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
args = SimpleNamespace(**read_yaml_file('../Conf/papyrus_config.yaml'))


#%%
args.delay = 1
args.modulation = 3
args.nLoop = 500

env = get_env(args)
env.gainCL = 0.9


env.atm.generateNewPhaseScreen(17)

data = np.zeros((args.nLoop // 50, len(np.arange(0, 0.4, 0.1))))

#%%
for alpha in np.arange(0, 0.4, 0.1):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    # savedir = '../../logs/'+args.savedir+'/integrator/'+f'{timestamp}'+'_'+args.experiment_tag+'_'+str(int(args.nLoop/args.frames_per_sec))+'s'+"_"+f'r0_{r0}_ws_{ws}_gain_{env.gainCL}'
    savedir = '../../logs/can_delete/integrator/'+f'{timestamp}'+'_'+args.experiment_tag+'_'+str(int(args.nLoop/args.frames_per_sec))+'s'+"_"+f'gain_{env.gainCL}'

    # print('Start make env')
    os.makedirs(savedir, exist_ok=True)

    # print("Running loop...")
    env.tel.resetOPD()
    env.dm.coefs = 0
    env.tel*env.dm*env.wfs


    env.atm.generateNewPhaseScreen(17)


    LE_PSFs= []
    SE_PSFs = []
    SRs = []
    SR_std = []
    rewards = []
    frames_for_pwr = []
    accu_reward = 0

    states = []

    use_integrator = False

    time_len = 500
    delay = 1

    n = 5
    m = 10
    obs4pred = np.zeros(n*m)
    buffer = np.zeros((n,m))
    pred = np.zeros((time_len - n + 1, 10))

    C2M = np.linalg.pinv(env.M2C_CL.copy())

    # alpha = 0.2


    obs = env.reset_soft()

    for i in range(args.nLoop):
        a=time.time()


        if i < 2:
            action = env.gainCL * obs 

        elif use_integrator:
            action = env.gainCL * obs 

        else:
            action = env.gainCL * ( alpha * env.vec_to_img(env.M2C_CL@pred_cmd_full) + (1-alpha) * obs.numpy())

        obs,_, reward,strehl, done, info = env.step(i,torch.tensor(action)) 

        modes_on_mirror = C2M@ ( env.dm.coefs.copy() * 1e6)

        residual_modes = - C2M@(env.img_to_vec(obs).numpy()) # Minus sign is because obs is negative

        pol_now = modes_on_mirror + residual_modes 

        states.append(pol_now)

        buffer = np.roll(buffer,1,axis=0)
        buffer[0] = pol_now[:m]

        # linear extrapolation
        pred_state = 2 * buffer[0] - buffer[1]

        pred_cmd = 2 * (buffer[0] - buffer[1])

        residual_modes[:m] = pred_cmd

        pred_cmd_full = - residual_modes

        if (i+1) >= n:
            pred[i - (n-1)] = buffer[0] + delay * (buffer[0] -  buffer[-1]) / (n-1) 

        # for EOF

        # for k in range(m):
        #     # in the k-th n block, we store the k-th mode's last n measurements
        #     # going from oldest to newest
        #     obs4pred[k * n: k* n + n] = buffer[:, k]

        accu_reward+= reward

        b= time.time()
        # print('Elapsed time: ' + str(b-a) +' s')

        # print('Loop '+str(i+1)+'/'+str(args.nLoop)+' Gain: '+str(env.gainCL)+' Turbulence: '+str(env.total[i])+' -- Residual:' +str(env.residual[i])+ '\n')
        # print("SR: " +str(strehl))
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


    data[:, int(alpha*10)] = np.array(SRs)
    # print("Data Saved")


    states = np.array(states)

    mode = 1
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
for i in range(data.shape[1]):
    plt.plot(data[:,i], label = f'alpha = {i/10}')
plt.legend()
plt.show()
# %%
