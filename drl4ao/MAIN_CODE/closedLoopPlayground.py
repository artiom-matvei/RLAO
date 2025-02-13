"""
Running the AO system in closed loop.
"""

#%%
import os,sys
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from ML_stuff.dataset_tools import read_yaml_file
from ML_stuff.models import Unet_big
import time
import numpy as np
from types import SimpleNamespace
import matplotlib.pyplot as plt
from PO4AO.mbrl import get_env

# Load in config file with simulation parameters
args = SimpleNamespace(**read_yaml_file('Conf/papyrus_config.yaml'))

# get_env will initialize the environment with our config parameters
env = get_env(args)


# One way to compute the Strehl Ratio is to compare the peak intensity
# of the PSF with no atmospheric turbulence (model PSF) to the peak
# intensity of the PSF you measure. Here we precompute the model PSF

env.tel.resetOPD()                  # Remove any turbulence
env.tel.computePSF(4)               # Compute PSF
psf_model_max = env.tel.PSF.max()   # Save PSF image

#%%
# Set your own save directory here
timestamp = time.strftime("%Y%m%d-%H%M%S")
savedir = '../../logs/'+args.savedir+'/integrator/'+args.experiment_tag+'_'+str(int(args.nLoop/args.frames_per_sec))+'s'
os.makedirs(savedir, exist_ok=True)

# Reset the state of the atmosphere
env.atm.generateNewPhaseScreen(17)

# Set all mirror actuators to position zero
# Save the shape of the mirror in prev_command
env.dm.coefs = 0
prev_command = env.dm.coefs.copy()

# Propagate the light from the telescope to mirror to the wavefront sensor
# Note that the atmosphere is coupled to the telescope already
env.tel*env.dm*env.wfs


LE_PSFs= []
SE_PSFs = []
SRs = []
SR_500 = []
SR_std = []
accu_reward = 0
LE_SR = []

# Here we start the control loop
for i in range(args.nLoop):
    a=time.time()

    # env.wfs.signal stores the most recent slope
    # calculation from the wfs
    wfs_signal = env.wfs.signal

    # The env.reconstructor is the interaction matrix that is 
    # created during the initialization of the env.
    # The IM takes the slope vector as input and
    # ouputs the corresponding DM actuator commands
    recontructed_phase = env.reconstructor @ wfs_signal

    # Rescale the action by a close-loop gain value
    # We multiply by -1 because we want to REMOVE the phase
    action = -1 * env.gainCL * recontructed_phase

    # Now that we compute the action, we will apply to the DM
    # We use a leaky integrator, this filters out drift.
    # Note that prev_command is in units of actuator position
    # and the action is in units of actuator displacement
    env.dm.coefs = (prev_command * env.leak) + action 
    prev_command = env.dm.coefs.copy()

    # Propagate the correction through the system
    env.tel*env.dm*env.wfs

    # At this point is where you would compute values
    # like the Strehl Ratio or the residual slope variance
    # or any metric you want to keep track of.

    # This is one way to compute the strehl but dont worry about the math
    strehl = np.exp(-np.var(env.tel.src.phase[np.where(env.tel.pupil==1)]))
    SR_500.append(strehl)

    # Update the atmosphere for the next iteration
    env.atm.update()

    b= time.time()
    print('Elapsed time: ' + str(b-a) +' s')

    print('Loop '+str(i+1)+'/'+str(args.nLoop)+' Gain: '+str(env.gainCL))
    print("SR: " +str(strehl) + "\n")

    # Here's an example of me wanting to compute some metrics
    # I wanted to save the average strehl over many frames  
    if (i+1) % 500 == 0:
        SRs.append(np.mean(SR_500))
        SR_std.append(np.std(SR_500))
        SR_500 = []


# Save the data if you want
print("Saving Data")
torch.save(SRs, os.path.join(savedir, "sr2plot.pt"))
torch.save(SR_std, os.path.join(savedir, "srstd2plot.pt"))
print("Data Saved")


# Possible exercises:
# - Adjust certain parameters like the gainCL to see how it affects performance / stability
# - Optimize the gainCL to maximize the Strehl Ratio
# - Animate the PSF over time (use env.tel.computePSF(n) and then env.tel.PSF to get PSF image [Experiment with n values])
# - Look into the env code to find the DM geometry and then visualize the DM shape over time (Hint: Look at the vec_to_img function in OOPAOEnv.py)
