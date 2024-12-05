# -*- coding: utf-8 -*-
"""
Playground for PO4AO
@author: Parker Levesque
git: @parks9
"""
# %%

# IMPORTS

import os,sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
import torch
from torch import optim
import numpy as np
from PO4AO.conv_models_simple import EnsembleDynamics, ConvPolicy
from PO4AO.util_simple import get_n_params, EfficientExperienceReplay
from ML_stuff.dataset_tools import read_yaml_file
from torch.utils.tensorboard.writer import SummaryWriter
from PO4AO.mbrl import get_env,run,train_dynamics,train_policy
from Plots.plots import save_plots
import time
from types import SimpleNamespace

#%%

# Import simulation configuration parameters
args = SimpleNamespace(**read_yaml_file('Conf/papyrus_config.yaml'))
args.delay = 1  # Set a frame delay for the environment

# Initialize the environment
env = get_env(args)

#%%

# Create a unique save directory for the experiment
timestamp = time.strftime("%Y%m%d-%H%M%S")
writer = SummaryWriter('../../logs/'+args.savedir+'/po4ao/'+f'{timestamp}'+'_'+args.experiment_tag+'_'+str(args.iters)+'s'+f'r0_{r0}_ws_{ws}')
savedir = '../../logs/'+args.savedir+'/po4ao/'+f'{timestamp}'+'_'+args.experiment_tag+'_'+str(args.iters)+'s'+f'r0_{args.r0}_ws_{args.ws}'
os.makedirs(savedir, exist_ok=True)



# Get filter from simulation that the agent will use
flt = env.F
flt = torch.from_numpy(np.asarray(flt)).float()

# Initialize the replay buffer, dynamics model, policy network and optimizers
replay = EfficientExperienceReplay((args.data_shape,args.data_shape), (args.data_shape,args.data_shape))
dynamics = EnsembleDynamics(env.xvalid, env.yvalid, args.n_history) 
policy = ConvPolicy(env.xvalid, env.yvalid, args.initial_sigma, flt, args.n_history)
dynamics_optimizer = optim.Adam(dynamics.parameters())
policy_optimizer = optim.Adam(policy.parameters())

# Make sure the device is set
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.gpu_device = device


# Initialize training variables

sigma = args.initial_sigma # Exploration noise in warm-up phase

reward_sums = torch.zeros(args.iters)
evals = torch.zeros(args.iters)
std = torch.zeros(args.iters)
past_obs = None
past_act = None

iteration = 0
obs = None


# Main training loop

for i in range(args.iters):
    
    start = time.time()

    # Run the policy to collect rollouts for the replay
    # and reward data to evaluate the performance
    
    strehl, reward_sum, past_obs, past_act, obs, rewards,iteration  = run(env, past_obs, past_act, obs,replay, policy, dynamics,args.n_history,args.max_ts,args.warmup_ts, sigma=sigma, writer=writer, episode = i,iteration=iteration)

    # Record results from this episode

    reward_sums[i] = reward_sum
    evals[i] =  strehl[0]
    std[i] = strehl[1]


    # Train the dynamics model and policy network

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
    
    if (i+1) % 10 == 0: # Save state every 10 iterations
        torch.save(dynamics.state_dict(), os.path.join(savedir, f"dynamics_{i+1}.pt"))
        torch.save(policy.state_dict(), os.path.join(savedir, f"policy_{i+1}.pt"))
        torch.save(rewards, os.path.join(savedir, "rewards.pt"))
        torch.save(evals, os.path.join(savedir, "evals.pt"))

    # Update exploration noise according to schedule

    sigma -= (args.initial_sigma / args.warmup_ts)
    sigma = max(0, sigma)



# Save rollout data at the end of the experiment
save_plots(savedir,evals,reward_sums,env.LE_PSF)

