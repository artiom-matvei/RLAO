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
import os
import matplotlib.pyplot as plt
from OOPAOEnv.OOPAOEnvRazor import OOPAO
import pandas as pd
from torch.utils.data import DataLoader


random.seed(5)
torch.manual_seed(5)
np.random.seed(5)


def get_env(args, gainCL=0.2, wfs_type='shackhartmann'):
    """Sets the OOPAO environment with the configuration file and wrappers it with torch.
    :return env: TorchWrapped OOPAO initialized environment.
    """
    env = OOPAO()
    env.set_params_file(args.param_file,args.oopao_path) # set parameter file
    env.set_params(args, wfs_type, gainCL=gainCL)   #sets env parameter file
    if args.delay < 0:
        env = TimeDelayEnv(env, args.delay)
    return TorchWrapper(env)

@torch.no_grad()
def run(env, past_obs, past_act, obs, replay, policy, dynamics,n_history,max_ts,warmup_ts, sigma, writer: SummaryWriter, episode,iteration): 
    """Run an entire Episode (based on max-ts = number of frames per Episode).
    :param: env, past_obs, past_act, obs, replay, policy, dynamics, sigma, writer: SummaryWriter, episode,iteration
    :return env.calculate_strehl_AVG(): Average Strehl ration in the Episode, float. 
    :return reward_sum: Sum of all the rewards in all the frames of the Episode, float. 
    :return past_obs: List of all past WFS measurements (states) through the Episode, list of matricex. 
    :return past_act: List of all past voltage DM commands (actions) through the Episode, list of matrices. 
    :return obs: Last WFS measurement (state) of the Episode, matrix (sized according to system parameters).
    :return rewards: List of all past rewards through the Episode, list of float.
    :return LE_PSF: Long Exposure PSF of the system after the current Episode, matrix/ image.
    """
    dynamics.eval()
    policy.eval()


    obs = env.reset_soft()

    reward_sum = 0
    rewards = []
    
    if past_obs == None:
            past_obs = torch.zeros(1, (n_history-1), *obs.shape).squeeze(2)
            past_act = torch.zeros(1, (n_history-1), *obs.shape).squeeze(2)
    
    for t in range(max_ts):
        simulated_obs = obs.unsqueeze(0)


        if  episode < warmup_ts:
            action = 0.5 * obs.unsqueeze(0)
            # action = action + torch.randn_like(action) * sigma
            action = action + env.sample_noise(sigma)
        else:            
            action = policy(simulated_obs.squeeze(0), torch.cat([past_obs, past_act],dim = 1), sigma = sigma)  
            action = action.squeeze(0)                                               
        
        next_obs, reward,strehl, done, _ = env.step(t,action.squeeze())
        
        #env.render()
        # LE_PSF,SE_PSF = env.render4plot(t)

        # roll telemetry data with new data
        past_obs = torch.cat([past_obs[:,1:,:,:], obs.unsqueeze(0).unsqueeze(0)], dim = 1) 
        past_act = torch.cat([past_act[:,1:,:,:], action.to(torch.float32).unsqueeze(0)], dim = 1)
        
        reward_sum += reward
        rewards.append(reward)

        action_to_save = action.squeeze().to(torch.float32)
        replay.append(obs, action_to_save, reward, next_obs, done)

        obs = next_obs
    
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
        #print(loss.item())
        loss.backward()

        opt.step()

    for p in dynamics.parameters():
        p.requires_grad_(True)

    policy.to('cpu')
    dynamics.to('cpu')

    return loss.item()


####################### Phase Reconstruction ############################

def OPD_model(dm_cmd, modes, res, xvalid, yvalid):

    vec_cmd = dm_cmd[:,:,xvalid, yvalid]
    dm_opd = torch.matmul(vec_cmd,modes.unsqueeze(0).unsqueeze(0).transpose(-1,-2)).squeeze(0)

    dm_opd = torch.reshape(dm_opd, (-1,res,res)).unsqueeze(1)

    return dm_opd


def make_diverse_dataset(env, size, num_scale=6):
    """Creates a pandas DataFrame with wavefront sensor measurements
    and corresponding mirror shapes, generated from normally distributed
    dm coefficients."""

    dataset = pd.DataFrame(columns=['wfs', 'dm'])


    sample_counter = 0

    scaling = np.logspace(-9, -6, num_scale)

    for i in range(num_scale):
        for j in range(size):

            print(f'scale factor:{scaling[i]}')

            command = np.random.randn(*env.dm.coefs.shape) * scaling[i]

            env.dm.coefs = command

            env.tel*env.dm*env.wfs

            dataset.loc[i*j + j] = {'wfs': env.wfs.cam.frame.copy(),\
                                     'dm': command}

            sample_counter += 1

            if j+1 == size:
                print(f"Generated {sample_counter} samples")

    return dataset



def get_OL_phase_dataset(env, size):
    """Creates a pandas DataFrame with wavefront sensor measurements
    and corresponding mirror shapes."""

    # Create random OPD maps
    tel_res = env.dm.resolution

    true_phase = np.zeros((tel_res,tel_res,size))

    dataset = pd.DataFrame(columns=['wfs', 'dm'])

    seeds = np.random.randint(1, 10000, size=size)

    for i in range(size):
        env.atm.generateNewPhaseScreen(seeds[i])
        env.tel*env.wfs

        dataset.loc[i] = {'wfs': np.array(env.wfs.cam.frame.copy()), 'dm': np.array(env.OPD_on_dm())}
        true_phase[:,:,i] = env.tel.OPD

        if i % 100 == 0:
            print(f"Generated {i} open loop samples")


    return true_phase, dataset

def get_CL_phase_dataset(env, size, reconstructor):
    """Creates a pandas DataFrame with wavefront sensor measurements
    and corresponding mirror shapes, using the closed loop system."""

    recontructor.eval()

    # Create random OPD maps

    tel_res = env.dm.resolution

    true_phase = np.zeros((tel_res,tel_res,size))

    dataset = pd.DataFrame(columns=['wfs', 'dm'])

    seeds = np.random.randint(1, 10000, size=size)

    for i in range(size):
        env.atm.generateNewPhaseScreen(seeds[i])
        env.tel*env.wfs

        obs = torch.tensor(env.wfs.cam.frame).clone().detach().float().unsqueeze(0).unsqueeze(0)

        with torch.no_grad(): 
            action = reconstructor(obs).squeeze(0).squeeze(0) #env.integrator()

        pred_OPD = OPD_model(action, env.dm.modes, env.dm.resolution, env.xvalid, env.yvalid)

        residual_phase = env.tel.OPD.copy() - pred_OPD.squeeze().numpy()

        env.tel.OPD = residual_phase
        env.tel*env.wfs

        dataset.loc[i] = {'wfs': np.array(env.wfs.cam.frame.copy()), 'dm': np.array(env.OPD_on_dm())}
        true_phase[:,:,i] = env.tel.OPD

        if i % 100 == 0:
            print(f"Generated {i} closed loop samples")

    return true_phase, dataset


def train_reconstructor(reconstructor, OPD_model, dataset, optimizer, criterion, n_epochs):
    """Trains the phase reconstructor model on the dataset of wavefront sensor measurements
    and corresponding mirror shapes."""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    reconstructor.train()
    reconstructor.to(device)

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    for epoch in range(n_epochs):
        running_loss = 0.0
        for inputs, targets in dataloader:
            # Zero the parameter gradients
            optimizer.zero_grad()
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = reconstructor(inputs)

            # Get the OPD from the model
            dm_OPD = OPD_model(outputs)

            loss = criterion(dm_OPD, targets)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {running_loss/len(dataloader)}")

    reconstructor.to('cpu')

    return loss.item()