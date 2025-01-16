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
                 gamma=0.1, tau=0.005, alpha=0.2, lr=1e-4, batch_size=256):
        self.env = env
        self.policy_net = policy_net
        self.q_net = q_net
        self.target_q_net = target_q_net
        self.replay_buffer = replay_buffer
        self.gamma = gamma
        self.tau = tau
        self.static_alpha = alpha
        self.batch_size = batch_size

        self.log_alpha = torch.tensor(np.log(alpha), requires_grad=True)  # Learnable log_alpha
        self.alpha = self.log_alpha.exp()  # alpha
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=1e-2)
        self.target_entropy = -357  # Target entropy, usually -dim(action space)


        # Optimizers
        self.q_optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr, weight_decay=1e-5)

        # Loss function
        self.criterion = nn.MSELoss()

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return None, None, None, None

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

            # Update alpha (entropy coefficient)
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp()  # Update alpha


        # Soft update of target Q-network
        with torch.no_grad():
            for target_param, param in zip(self.target_q_net.parameters(), self.q_net.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return q_loss, policy_loss, current_q_values, actions

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
            step_count = 0
            for step in range(max_steps):
                action = self.select_action(state)
                next_state, reward, done, *_ = self.env.step(action)
                self.replay_buffer.add(state, action, reward, next_state, done)
                state = next_state
                episode_reward += reward

                step_count += 1

                if step % update_frequency == 0:
                    q_loss, policy_loss, current_q_values, actions = self.update()

                    if q_loss is not None:
                        print(f"  Q-Loss: {q_loss.item()}")
                        print(f"  Policy Loss: {policy_loss.item()}")
                        print(f"  Mean Q-value: {current_q_values.mean().item()}")
                        rms_actions = torch.sqrt(torch.mean(actions.square()))
                        print(f"  Actions Mean/Std: {rms_actions.item()}/{actions.std().item()}")
                        print(f"  Alpha: {self.alpha.item()}")
                    else:
                        print("  Not enough samples in the replay buffer")

                if done:
                    break

            rewards.append(episode_reward / step_count)
            print(f"Episode {episode}:")
            print(f"  Mean reward: {episode_reward / step_count}")



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
    def __init__(self, state_dim, nValidAct, xvalid, yvalid, gainCL=0.5):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(state_dim[0], 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)
        )

        self.log_std = nn.Parameter(torch.ones(nValidAct) * 1e-6)  # Learnable log_std
        self.delta_scale = nn.Parameter(torch.tensor(1e-3))


        self.xvalid = xvalid
        self.yvalid = yvalid
        self.gainCL = gainCL


    def sample(self, state):

        mean2D = self.fc(state).squeeze(1)
        meanVec = mean2D[:, self.xvalid, self.yvalid]

        log_std = self.log_std.clamp(min=-20, max=2).expand_as(meanVec)
        std = torch.exp(log_std).clamp(min=1e-6)
        dist = torch.distributions.Normal(meanVec, std)

        delta_action = dist.rsample()
        delta_action = torch.clamp(delta_action, min=-1e-4, max=1e-4)
        log_prob = dist.log_prob(delta_action).sum(axis=-1, keepdims=True)

        base_action = state[:, 0, self.xvalid, self.yvalid]
        action = (base_action + self.delta_scale * delta_action) * self.gainCL

        return action, log_prob
    
    def forward(self, state):
        mean2D = self.fc(state).squeeze(1)
        meanVec = mean2D[:, self.xvalid, self.yvalid]

        base_action = state[:, 0, self.xvalid, self.yvalid]

        return (base_action + self.delta_scale * meanVec) * self.gainCL 




class QNet(nn.Module):
    def __init__(self, state_dim, nValidAct, xvalid, yvalid):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(state_dim[0] + 1 , 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)
        )
        self.q = nn.Linear(nValidAct, 1)

        self.xvalid = xvalid
        self.yvalid = yvalid

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        out = self.fc(x).squeeze(1)
        q_vals = self.q(out[:, self.xvalid, self.yvalid])

        return q_vals