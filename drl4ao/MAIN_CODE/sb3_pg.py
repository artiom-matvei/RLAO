# %%
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import gym
import os,sys
from OOPAOEnv.OOPAOEnv_VPG import OOPAO
from VPG.SAC import ReplayBuffer, SACAgent, PolicyNet, QNet
#%%
env = OOPAO()
state_dim = env.observation_space.shape
action_dim = env.action_space.shape
nValidAct = env.dm.nValidAct
#%%

replay_buffer = ReplayBuffer(capacity=100000)
policy_net = PolicyNet(state_dim, nValidAct, env.xvalid, env.yvalid)
q_net = QNet(state_dim, nValidAct, env.xvalid, env.yvalid)
target_q_net = QNet(state_dim, nValidAct, env.xvalid, env.yvalid)
target_q_net.load_state_dict(q_net.state_dict())


# Warmup the agent
# agent.warmup(num_steps=int(1e4))

def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

policy_net.apply(init_weights)
q_net.apply(init_weights)
target_q_net.apply(init_weights)

agent = SACAgent(env, policy_net, q_net, target_q_net, replay_buffer, alpha=0.0001)

rewards = agent.train(num_episodes=200, max_steps=250, update_frequency=100)


np.save("rewards.npy", np.array(rewards))
# %%
# obs, _  = env.reset()
# reward_sum = 0
# ep_len = 200
# for i in range(ep_len):
#     # action = agent.select_action(obs)
#     action = 0.5 * obs[0]
#     obs, reward, done, *_ = env.step(action)
#     reward_sum += reward
#     # env.render("human")

# print(f"Mean reward: {reward_sum/ep_len}")
# %%
