# -*- coding: utf-8 -*-
"""
OOPAO module for PO4AO
@author: Raissa Camelo (LAM) git: @srtacamelo
"""

# %%
import os,sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
import torch
#torch.set_deterministic(True)
from torch import optim
import numpy as np
from PO4AO.conv_models_simple import EnsembleDynamics, ConvPolicy
from PO4AO.util_simple import get_n_params, EfficientExperienceReplay
from ML_stuff.dataset_tools import read_yaml_file
from torch.utils.tensorboard.writer import SummaryWriter

# For RAZOR sim
# from PO4AO.mbrl_funcsRAZOR import get_env,run,train_dynamics,train_policy

#For Papyrus sim
from PO4AO.mbrl import get_env,run,train_dynamics,train_policy

from Plots.plots import save_plots
import time

# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# Parser
from Conf.parser_Configurations import Config, ConfigAction
import argparse
from types import SimpleNamespace
import matplotlib.pyplot as plt


#%%
# args = SimpleNamespace(**read_yaml_file('Conf/razor_config_po4ao.yaml'))
args = SimpleNamespace(**read_yaml_file('Conf/papyrus_config.yaml'))


# %%

# if __name__=='__main__':

args.delay = 1

#%%

for r0 in [0.13, 0.0866666667]:
    args.r0 = r0
    env = get_env(args)

    for ws in [[10,12,11,15,20], [20,24,22,30,40]]:
        env.atm.windSpeed = ws
        env.atm.generateNewPhaseScreen(17)

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        writer = SummaryWriter('../../logs/'+args.savedir+'/po4ao/'+f'{timestamp}'+'_'+args.experiment_tag+'_'+str(args.iters)+'s'+f'r0_{r0}_ws_{ws}')
        savedir = '../../logs/'+args.savedir+'/po4ao/'+f'{timestamp}'+'_'+args.experiment_tag+'_'+str(args.iters)+'s'+f'r0_{r0}_ws_{ws}'


        os.makedirs(savedir, exist_ok=True)
        # args.save(savedir+"/arguments"+'_.json')

        """Main function that initiates the enviroment, sets up the policy and the dynamics model neural networks (NN).
        Contains the experiment main loop, running the system and the training phase for each NN.
        Saves training states and results.
        :return: evals,reward_sums,env.LE_PSF
        """
        # env = get_env(args)
        # env.change_mag(4)



        flt = env.F
        flt = torch.from_numpy(np.asarray(flt)).float()

        replay = EfficientExperienceReplay((args.data_shape,args.data_shape), (args.data_shape,args.data_shape))

        dynamics = EnsembleDynamics(env.xvalid, env.yvalid, args.n_history) 
        policy = ConvPolicy(env.xvalid, env.yvalid, args.initial_sigma, flt, args.n_history)

        dynamics_optimizer = optim.Adam(dynamics.parameters())
        policy_optimizer = optim.Adam(policy.parameters())
        # device = torch.device(args.gpu_device)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        args.gpu_device = device

        #print('params', get_n_params(dynamics), get_n_params(policy))

        sigma = args.initial_sigma


        reward_sums = torch.zeros(args.iters)
        evals = torch.zeros(args.iters)
        std = torch.zeros(args.iters)
        past_obs = None
        past_act = None

        iteration = 0
        obs = None

        for i in range(args.iters):
            
            start = time.time()
            
            strehl, reward_sum, past_obs, past_act, obs, rewards,iteration  = run(env, past_obs, past_act, obs,replay, policy, dynamics,args.n_history,args.max_ts,args.warmup_ts, sigma=sigma, writer=writer, episode = i,iteration=iteration)

            if reward_sum < -46:
                converged = 1
            reward_sums[i] = reward_sum
            evals[i] =  strehl[0]
            std[i] = strehl[1]

            dyn_loss = 0
            pol_loss = 0

            if i == args.warmup_ts -1: # During warm-up phase
                dyn_loss = train_dynamics(args.n_history,args.max_ts,args.batch_size,dynamics, dynamics_optimizer, replay, dyn_iters=100,device=args.gpu_device)
                # Ensure all operations on GPU are finished before proceeding
                if args.gpu_device == 'cuda':
                    torch.cuda.synchronize()
                pol_loss = train_policy(policy_optimizer, policy, dynamics, replay,args.gpu_device,args.n_history, args.max_ts, args.batch_size,args.T, pol_iters=60)
            if i > args.warmup_ts -1: # After warm-up phase, until the end of the experiment
                dyn_loss = train_dynamics(args.n_history,args.max_ts,args.batch_size,dynamics, dynamics_optimizer, replay, dyn_iters=10,device=args.gpu_device)
                # Ensure all operations on GPU are finished before proceeding
                if args.gpu_device == 'cuda':
                    torch.cuda.synchronize()
            
                pol_loss = train_policy(policy_optimizer, policy, dynamics, replay,args.gpu_device,args.n_history, args.max_ts, args.batch_size,args.T, pol_iters=7)
                # Ensure all operations on GPU are finished before proceeding
                if args.gpu_device == 'cuda':
                    torch.cuda.synchronize()

            # Save training results for this Episode

            writer.add_scalar('train/pol_loss', pol_loss, i)          
            writer.add_scalar('train/dyn_loss', dyn_loss, i)      
            writer.add_scalar('train/strehl', strehl[0], i)
            writer.add_scalar('train/reward_sum', reward_sum, i)
            print(f'Iteration (Episode) {i} complete ({time.time() - start:.2f}s). \n\t dyn:{dyn_loss:.4f} pol:{pol_loss:.4f} \n\t strehl:{strehl[0]:.3f} reward:{reward_sum:.3f}')
            
            sigma -= (args.initial_sigma / args.warmup_ts)
            sigma = max(0, sigma)

            if (i+1) % 10 == 0: # Save state every 10 iterations
                torch.save(dynamics.state_dict(), os.path.join(savedir, f"dynamics_{i+1}.pt"))
                torch.save(policy.state_dict(), os.path.join(savedir, f"policy_{i+1}.pt"))
                torch.save(rewards, os.path.join(savedir, "rewards.pt"))
                torch.save(evals, os.path.join(savedir, "evals.pt"))

        # env.render4plot(15) # PSF images

        print("Saving Data")
        save_plots(savedir,evals,reward_sums,env.LE_PSF) #
        print("Data Saved")
# %%
