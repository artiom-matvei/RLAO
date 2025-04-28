#%%
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import random
import time
import matplotlib.pyplot as plt
import gymnasium as gym
import numpy as np
import torch

from OOPAOEnv.learnIMEnv import OOPAO
from learnIM import Actor

env = OOPAO()

def make_env():
    def thunk():
        env = OOPAO()
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    return thunk

envs = gym.vector.SyncVectorEnv([make_env()])

actor = Actor(envs)

actor.load_state_dict(torch.load("./models/best_model.pth", map_location=torch.device('cpu'))["model_state_dict"])
# %%
# Residuals with turbulence

# Reset the environment
obs, info = env.reset(seed=0)


residuals_turbulence = []
# Start the loop
for i in range(env.args.nLoop):
    # Take a step in the environment
    obs, reward, terminated, truncated, info = env.step(np.zeros_like(obs))

    residuals_turbulence.append(obs[0])

    if (i + 1) % 100 == 0:
        print(f"Step {i + 1}/{env.args.nLoop}")

# Residuals with integrator

# Reset the environment
obs, info = env.reset(seed=0)

residuals_integrator = []
# Start the loop
for i in range(env.args.nLoop):
    #Integrator action
    action = -1 * obs
    # Take a step in the environment
    obs, reward, terminated, truncated, info = env.step(action)

    residuals_integrator.append(obs[0])

    if (i + 1) % 100 == 0:
        print(f"Step {i + 1}/{env.args.nLoop}")

# Residuals with actor

obs, info = env.reset(seed=0)
residuals_actor = []

# Start the loop
for i in range(env.args.nLoop):
    # Take a step in the environment
    actions, _, _ = actor.get_action(torch.Tensor(obs[np.newaxis, :]))

    obs, reward, terminated, truncated, info = env.step(action)

    residuals_actor.append(obs[0])

    if (i + 1) % 100 == 0:
        print(f"Step {i + 1}/{env.args.nLoop}")



# %%

x = np.arange(env.args.nLoop)
rl = residuals_actor 
integrator = residuals_integrator
no_correction = residuals_turbulence

# Create the figure and subplots
fig, axs = plt.subplots(3, 1, figsize=(10, 5), sharex=True)

# RL agent
axs[0].plot(x, rl, color='orangered', linewidth=2)
axs[0].axhline(0, color='black', linestyle='--', linewidth=1.5)
axs[0].set_title("RL agent")
axs[0].set_ylim(-0.2, 0.2)
axs[0].set_xlim(0, env.args.nLoop)

# Integrator
axs[1].plot(x, integrator, color='navy', linewidth=2)
axs[1].axhline(0, color='black', linestyle='--', linewidth=1.5)
axs[1].set_title("Integrator")
axs[1].set_ylim(-0.2, 0.2)
axs[1].set_xlim(0, env.args.nLoop)

# No correction
axs[2].plot(x, no_correction, color='black', linewidth=2)
axs[2].axhline(0, color='black', linestyle='--', linewidth=1.5)
axs[2].set_title("No correction")
axs[2].set_ylim(-0.2, 0.2)
axs[2].set_xlim(0, env.args.nLoop)

# Shared y-axis label
fig.text(0.04, 0.5, r'Residuals $(\lambda/D)$', va='center', rotation='vertical', fontsize=12)

# Shared x-axis label
plt.xlabel("Iteration")

plt.tight_layout(rect=[0.05, 0.05, 1, 0.97])
plt.show()

# %%

from scipy.signal import welch

# Sampling frequency in Hz (adjust this to match your real system)
fs = 500  # For example, 1000 Hz

# Calculate the Power Spectral Density using Welch's method
f_rl, psd_rl = welch(rl, fs=fs, nperseg=256)
f_int, psd_int = welch(integrator, fs=fs, nperseg=256)
f_nc, psd_nc = welch(no_correction, fs=fs, nperseg=256)

# Plot the PSDs
plt.figure(figsize=(7, 5))
plt.loglog(f_rl, psd_rl, label="RL", color="orangered", linewidth=2)
plt.loglog(f_int, psd_int, label="Integrator", color="navy", linewidth=2)
plt.loglog(f_nc, psd_nc, label="No correction", color="black", linewidth=2)

plt.xlabel("Frequency (Hz)")
plt.ylabel("Power Spectral Density")
plt.legend()
# plt.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()
plt.show()

# %%
