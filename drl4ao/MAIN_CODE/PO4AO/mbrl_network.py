# -*- coding: utf-8 -*-
"""
OOPAO module for PO4AO 
@author: Raissa Camelo (LAM) git: @srtacamelo
"""
import torch
#torch.set_deterministic(True)
from PO4AO.util import Dynamics
from torch import optim
import numpy as np
from PO4AO.util_simple import  TorchWrapper, EfficientExperienceReplay, TimeDelayEnv
from torch.utils.tensorboard.writer import SummaryWriter
import random
import matplotlib.pyplot as plt
from OOPAOEnv.OOPAOEnv import OOPAO


random.seed(5)
torch.manual_seed(5)
np.random.seed(5)

def get_env(args):
    """Sets the OOPAO environment with the configuration file and wrappers it with torch.
    :return env: TorchWrapped OOPAO initialized environment.
    """
    env = OOPAO()
    env.set_params_file(args.param_file,args.oopao_path) # set parameter file
    env.set_params(args)   #sets env parameter file
    if args.delay < 0:
        env = TimeDelayEnv(env, args.delay)
    return TorchWrapper(env)

@torch.no_grad()
def run(env, past_obs, past_act, obs, replay, policy, dynamics,n_history,max_ts,warmup_ts, sigma, writer: SummaryWriter, episode,iteration, reconstructor, c_int): 
    """Run an entire Episode (based on max-ts = number of frames per Episode).
    :param: env, past_obs, past_act, obs, replay, policy, dynamics, sigma, writer: SummaryWriter, episode,iteration
    :return env.calculate_strehl_AVG(): Average Strehl ration in the Episode, float. 
    :return reward_sum: Sum of all the rewards in all the frames of the Episode, float. 
    :return past_obs: List of all past WFS measurements (states) through the Episode, list of matricex. 
    :return past_act: List of all past voltage DM commands (actions) through the Episode, list of matrices. 
    :return obs: Last WFS measurement (state) of the Episode, matrix (sized according to system parameters).
    :return rewards: List of all past rewards through the Episode, list of float.
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dynamics.eval()
    policy.eval()

    policy.to(device)

    c_net = 1 - c_int

   
    env.atm.generateNewPhaseScreen(93234 * iteration)
    env.dm.coefs = 0

    env.tel*env.dm*env.wfs


    # Get the initial observations
    im = env.reset_soft()
    wfsf = torch.tensor(env.wfs.cam.frame.copy()).float().unsqueeze(1).to(device)

    int_action = env.gainCL * im.unsqueeze(0).to(device)

    reshaped_input = wfsf.view(-1, 2, 24, 2, 24).permute(0, 1, 3,2, 4).contiguous().view(-1, 4, 24, 24)
    with torch.no_grad():
        tensor_output = reconstructor(reshaped_input).squeeze()
        net_action = env.net_gain * torch.sinh(tensor_output)
    
    obs = c_int * int_action - c_net * net_action

    obs.to(device)

    reward_sum = 0
    rewards = []
    
    if past_obs == None:
            past_obs = torch.zeros(1, (n_history-1), *obs.shape).squeeze(2).to(device)
            past_act = torch.zeros(1, (n_history-1), *obs.shape).squeeze(2).to(device)

    for t in range(max_ts):

        simulated_obs = obs / env.gainCL

        if  episode < warmup_ts:
            # action = env.gainCL * obs.unsqueeze(0)
            # action = action + torch.randn_like(action) * sigma
            action = obs + env.sample_noise(sigma, use_torch=True).to(device)
        else:            
            action = policy(simulated_obs.squeeze(0), torch.cat([past_obs, past_act],dim = 1))  
            action = action.squeeze(0)                                               
        
        im, wfsf, reward,strehl, done, _ = env.step(t,action.squeeze())

        # Craft the next observation
        int_action = env.gainCL * im.unsqueeze(0).to(device)
        wfsf = torch.tensor(wfsf).float().unsqueeze(1).to(device)

        reshaped_input = wfsf.view(-1, 2, 24, 2, 24).permute(0, 1, 3,2, 4).contiguous().view(-1, 4, 24, 24)
        with torch.no_grad():
            tensor_output = reconstructor(reshaped_input).squeeze()
            net_action = torch.sinh(tensor_output)  # Now convert to NumPy
        
        next_obs = c_int * int_action - c_net * net_action

        next_obs.to(device)

        # roll telemetry data with new data
        past_obs = torch.cat([past_obs[:,1:,:,:], obs.unsqueeze(0).to(device)], dim = 1) 
        past_act = torch.cat([past_act[:,1:,:,:], action.to(torch.float32).unsqueeze(0).to(device)], dim = 1)


        reward_sum += reward
        rewards.append(reward)

        action_to_save = action.squeeze().to(torch.float32).to(device)
        replay.append(obs, action_to_save, reward, next_obs, done)

        obs = next_obs.to(device)
    
    return env.calculate_strehl_AVG(), reward_sum, past_obs, past_act, obs, rewards,iteration


def train_dynamics(n_history,max_ts,batch_size,dynamics: Dynamics, optimizer: optim.Adam, replay: EfficientExperienceReplay, dyn_iters=5,device ='cuda:0'):
    """Trains the dynamics model on the data saved from the last Episode ([state, action, next state] for each iteraction in the Episode).
    Saves the trained model to CPU.
    This model predicts the next state (next WFS measure) based on the previous state and action (voltages applied to the DM).
    :return loss: Loss of the model after training.
    """
    dynamics.train()
    dynamics.to(device)

    for i in range(dyn_iters):
        optimizer.zero_grad()
        
        loss = 0

        for bs_model in dynamics.models:
            sample = replay.sample_contiguous(n_history, max_ts, batch_size)
            
            states = sample.state()
            actions = sample.action()

            states = states.view(batch_size, n_history + 1, 1, *states.shape[1:]).to(device)
            states_unfolded = states[:, :-1]
            
            actions_unfolded = actions.view(batch_size, n_history + 1, *actions.shape[1:]).to(device)
            actions_unfolded = actions_unfolded[:, :-1].unsqueeze(2)
            
            next_states = states[:, -1]

            state = states_unfolded[:,-1].squeeze(2) 
            action = actions_unfolded[:,-1].squeeze(2)  

            history = torch.cat([states_unfolded[:,:-1].squeeze(2), actions_unfolded[:,:-1].squeeze(2)], dim=1) 

            pred = bs_model(state, action, history)

            assert pred.shape == next_states.shape
            pred_loss = (next_states - pred).pow(2).mean()
            
            loss += pred_loss

        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(dynamics.parameters(), 0.5)

        optimizer.step()

    dynamics.to('cpu')

    return loss.item()


def loss_fn(state, action):
    return state.pow(2).mean() + 0.001*action.pow(2).mean()

def train_policy(opt, policy, dynamics, replay,device,n_history, max_ts, batch_size,T, pol_iters=5):
    """Trains the policy model on the data saved from the last Episode ([state, action, next state] for each iteraction in the Episode) and using the pre-trained Dynamics model.
    Saves the trained model to CPU.
    This model predicts the next action (voltages to apply to the DM) based on the current state (current WFS measure).
    :return loss: Loss of the model after training.
    """
    dynamics.train()
    policy.train()

    for p in dynamics.parameters():
        p.requires_grad_(False)

    policy.to(device)
    dynamics.to(device)

    for i in range(pol_iters):
        opt.zero_grad()
        
        sample = replay.sample_contiguous(n_history, max_ts, batch_size).to(device)

        b = len(sample)
        
        states = sample.state()
        actions = sample.action()

        states_unfolded = states.view(batch_size, n_history + 1, 1, *states.shape[1:]).to(device)
        states_unfolded = states_unfolded[:, :-1]
        
        actions_unfolded = actions.view(batch_size, n_history + 1, *actions.shape[1:]).to(device)
        actions_unfolded = actions_unfolded[:, :-1].unsqueeze(2)        

        state = states_unfolded[:,-1].squeeze(2) 
        action = actions_unfolded[:,-1].squeeze(2)  

        # get past telemetry data
        past_obs = states_unfolded[:,:-1].squeeze(2)
        past_act = actions_unfolded[:,:-1].squeeze(2)  

        losses = torch.zeros(b, device=device)

        for t in range(0, T):

            history = torch.cat([past_obs, past_act], dim=1)

            if n_history > 1:
                action = policy(state, history)
                next_state = dynamics(state, action, history)           
            else:
                action = policy(state)
                next_state = dynamics(state, action)
            
            
            losses += loss_fn(next_state[:, 0],action)

            # roll history
            past_act = torch.cat([past_act[:,1:,:,:], action], dim = 1) 
            past_obs = torch.cat([past_obs[:,1:,:,:], state], dim = 1)

            
            next_state = torch.mean(next_state, dim = 1, keepdim = True)
            state = next_state

        loss = losses.mean()
        loss.backward()

        opt.step()

    for p in dynamics.parameters():
        p.requires_grad_(True)

    policy.to('cpu')
    dynamics.to('cpu')

    return loss.item()
