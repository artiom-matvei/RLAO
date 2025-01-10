import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import gym
import os,sys

# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        states, actions, rewards, next_states, dones = zip(*random.sample(self.buffer, batch_size))
        return (np.array(states), np.array(actions), np.array(rewards), 
                np.array(next_states), np.array(dones))

    def __len__(self):
        return len(self.buffer)

# Soft Actor-Critic Agent
class SACAgent:
    def __init__(self, env, policy_net, q_net, target_q_net, replay_buffer, 
                 gamma=0.99, tau=0.005, alpha=0.2, lr=3e-4, batch_size=256):
        self.env = env
        self.policy_net = policy_net
        self.q_net = q_net
        self.target_q_net = target_q_net
        self.replay_buffer = replay_buffer
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.batch_size = batch_size

        # Optimizers
        self.q_optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        # Loss function
        self.criterion = nn.MSELoss()

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        # Sample from the replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        # Update Q-network
        with torch.no_grad():
            next_actions, next_log_probs = self.policy_net.sample(next_states)
            # Reshape 1d action vector into 2d action map
            try:
                next_action_reshape = self.action_reshape(next_actions, next_states)
            except:
                next_action_reshape = next_actions.unsqueeze(1)

            target_q_values = self.target_q_net(next_states, next_action_reshape)
            target = rewards + self.gamma * (1 - dones) * (target_q_values - self.alpha * next_log_probs)
        
        try:
            actions_reshape = self.action_reshape(actions, states)
        except:
            actions_reshape = actions.unsqueeze(1)

        current_q_values = self.q_net(states, actions_reshape)
        q_loss = self.criterion(current_q_values, target)

        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # Update policy network
        actions, log_probs = self.policy_net.sample(states)
        actions_reshape = self.action_reshape(actions, states)
        q_values = self.q_net(states, actions_reshape)
        policy_loss = (self.alpha * log_probs - q_values).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Soft update of target Q-network
        for target_param, param in zip(self.target_q_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def select_action(self, state, deterministic=False):
        state = torch.FloatTensor(state).unsqueeze(0)
        action, _ = self.policy_net.sample(state)
        if deterministic:
            return action.detach().cpu().numpy()[0]
        return action.detach().cpu().numpy()[0]

    def train(self, num_episodes, max_steps, update_frequency=1):
        rewards = []
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            for step in range(max_steps):
                action = self.select_action(state)
                next_state, reward, done, *_ = self.env.step(action)
                self.replay_buffer.add(state, action, reward, next_state, done)
                state = next_state
                episode_reward += reward

                if step % update_frequency == 0:
                    self.update()

                if done:
                    break

            rewards.append(episode_reward)
            print(f"Episode {episode + 1}, Reward: {episode_reward}")

        return rewards
    
    def warmup(self, num_steps):
        state, _ = self.env.reset()
        for i in range(num_steps):
            action = self.env.gainCL * state[0]
            vec_action = action[self.env.xvalid, self.env.yvalid]
            next_state, reward, done, *_ = self.env.step(action)
            self.replay_buffer.add(state, vec_action, reward, next_state, done)
            state = next_state
            if done:
                state = self.env.reset()

            if (i + 1)%1000 == 0:
                print(f"Step {i + 1}")

    
    def action_reshape(self, actions, states):
        action_reshape = torch.zeros((states.shape[0], states.shape[2], states.shape[3]))
        action_reshape[:, self.env.xvalid, self.env.yvalid] = actions
        return action_reshape.unsqueeze(1)
    


    # Define your policy network and Q-network
class PolicyNet(nn.Module):
    def __init__(self, state_dim, nValidAct, xvalid, yvalid):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(state_dim[0], 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
        )

        self.log_std = nn.Parameter(torch.zeros(nValidAct))  # Learnable log_std

        self.xvalid = xvalid
        self.yvalid = yvalid


    def sample(self, state):
        mean2D = self.fc(state).squeeze(1)
        meanVec = mean2D[:, self.xvalid, self.yvalid]

        log_std = self.log_std.expand_as(meanVec).clamp(min=-20, max=2)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(meanVec, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(axis=-1, keepdims=True)
        return action, log_prob

class QNet(nn.Module):
    def __init__(self, state_dim, nValidAct, xvalid, yvalid):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(state_dim[0] + 1 , 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
        )
        self.q = nn.Linear(nValidAct, 1)

        self.xvalid = xvalid
        self.yvalid = yvalid

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        out = self.fc(x).squeeze(1)
        q_vals = self.q(out[:, self.xvalid, self.yvalid])

        return q_vals