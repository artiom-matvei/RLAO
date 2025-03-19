#%%
import os, sys
import numpy as np
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ML_stuff.dataset_tools import read_yaml_file
from types import SimpleNamespace
from PO4AO.mbrl import get_env


try:
    args = SimpleNamespace(**read_yaml_file('./Conf/papyrus_config.yaml'))
except:
    args = SimpleNamespace(**read_yaml_file('../Conf/papyrus_config.yaml'))

env = get_env(args)

#%%
savedir = os.path.dirname(__file__) + '/../diffusion/images/thesis_data'

if not os.path.exists(savedir):
    os.makedirs(savedir)

#%%
size = int(1e3)


# dm_commands = np.memmap(savedir+f'/dm_cmds.npy', dtype='float32', mode='w+', \
#                                             shape=(size, *env.dm.coefs.shape))
# wfs_frames = np.memmap(savedir+f'/wfs_frames_atm.npy', dtype='float32', mode='w+', \
#                                                 shape=(size, *env.wfs.cam.frame.shape))
# wfs_frames_filtered = np.memmap(savedir+f'/wfs_frames_filtered.npy', dtype='float32', mode='w+', \
#                                                 shape=(size, *env.wfs.cam.frame.shape))

lr = np.zeros((1000, 4, 24, 24))
hr = np.zeros((1000, 4, 24, 24))

m2opd = np.load(os.path.dirname(__file__)+'/M2OPD_500modes.npy')
opd2m = np.linalg.pinv(m2opd)
m2c   = np.load(os.path.dirname(__file__)+'/M2C_357modes.npy')
xpupil, ypupil = np.where(env.tel.pupil == 1)



start = time.time()

for j in range(size):
    
    # Random Atmospheric Phase Screen
    env.atm.generateNewPhaseScreen(np.random.randint(0, 1e8))
    env.tel*env.wfs

    # Save the WFS frame
    # wfs_frames[j] = np.float32(env.wfs.cam.frame.copy())

    # Split into four 24x24 quadrants
    frame = env.wfs.cam.frame.copy()
    q1 = frame[:24, :24]  # Top-left
    q2 = frame[:24, 24:]  # Top-right
    q3 = frame[24:, :24]  # Bottom-left
    q4 = frame[24:, 24:]  # Bottom-right

    # Stack them along a new channel dimension
    hr[j] = np.float32(np.stack([q1, q2, q3, q4], axis=0))


    # Zernike Modes of Phase
    phase = env.tel.OPD.copy()
    modes = opd2m@phase[xpupil, ypupil]

    # Save nModes controllable modes on DM as Grount Truth Commands
    nModes = 30
    # dm_commands[j] = np.float32(m2c[:, :nModes]@modes[:nModes])

    # Filter Phase to nModes modes
    filteredPhase = np.zeros_like(phase)
    filteredPhase[xpupil, ypupil] = m2opd[:,:nModes]@modes[:nModes]

    # Propagate Filtered Phase to WFS
    env.tel.OPD = filteredPhase.copy()
    env.tel*env.wfs

    # Save the WFS frame
    # wfs_frames_filtered[j] = np.float32(env.wfs.cam.frame.copy())
    # Split into four 24x24 quadrants
    frame = env.wfs.cam.frame.copy()
    q1 = frame[:24, :24]  # Top-left
    q2 = frame[:24, 24:]  # Top-right
    q3 = frame[24:, :24]  # Bottom-left
    q4 = frame[24:, 24:]  # Bottom-right

    # Stack them along a new channel dimension
    lr[j] = np.float32(np.stack([q1, q2, q3, q4], axis=0))


    if (j+1) % 1000 == 0:
        with open("thesis_dataset.txt", "a") as f:
            f.write(f"Generated {j + 1} samples in {time.time()-start} seconds\n")
        start = time.time()


np.save(f'{savedir}/hr', hr)
np.save(f'{savedir}/lr', lr)
# dm_commands.flush()
# wfs_frames.flush()
# wfs_frames_filtered.flush()
# %%
