#%%
import os,sys
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from zernike import RZern
from scipy.optimize import curve_fit

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../AO_OOPAO')))

from OOPAO.tools.displayTools import displayMap
import time
import numpy as np
from types import SimpleNamespace
from ML_stuff.dataset_tools import read_yaml_file, data_from_stats
from ML_stuff.models import Reconstructor, Reconstructor_2, Unet_big
#For Razor sim
# from PO4AO.mbrl_funcsRAZOR import get_env
# try:
#     args = SimpleNamespace(**read_yaml_file('./Conf/razor_config_po4ao.yaml'))
# except:
#     args = SimpleNamespace(**read_yaml_file('../Conf/razor_config_po4ao.yaml'))
#For papyrus sim
from PO4AO.mbrl import get_env
from Plots.AO_plots import make_M2OPD, zernike_dist

try:
    args = SimpleNamespace(**read_yaml_file('./Conf/papyrus_config.yaml'))
except:
    args = SimpleNamespace(**read_yaml_file('../Conf/papyrus_config.yaml'))


import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'inferno'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# savedir = os.path.dirname(__file__)

env = get_env(args)

#%%

m2opd = np.load(os.path.dirname(__file__)+'/wf_recon/M2OPD_500modes.npy')
opd2m = np.linalg.pinv(m2opd)
m2c   = np.load(os.path.dirname(__file__)+'/wf_recon/M2C_357modes.npy')
xpupil, ypupil = np.where(env.tel.pupil == 1)

path = '/home/parker09/projects/def-lplevass/parker09/RLAO/logs/can_delete/integrator/20240926-093309_test_40s_r0_0.13_ws_[10, 12, 11, 15, 20]_gain_0.9/OPD_frames_500.npy'

pwr_OL, fit_OL = zernike_dist(env, m2opd, full_fit=True)

# pwr_CL, fit_CL = zernike_dist(env, M2OPD, path)

# mid = np.logspace(np.log10(fit_CL), np.log10(fit_OL), 3)
# %%
tag = 'thesis'
size = 400000

savedir = '/home/parker09/projects/def-lplevass/parker09/RLAO/drl4ao/MAIN_CODE/diffusion/datasets'

# dm_commands = np.memmap(savedir+f'/dm_cmds_{tag}.npy', dtype='float32', mode='w+', \
#                                             shape=(size, *env.dm.coefs.shape))
# wfs_frames = np.memmap(savedir+f'/wfs_frames_{tag}.npy', dtype='float32', mode='w+', \
#                                             shape=(size, *env.wfs.cam.frame.shape))
wfs_LO = np.memmap(savedir+f'/wfs_LO_{tag}.npy', dtype='float32', mode='w+', \
                                              shape=(size, *env.wfs.cam.frame.shape))

wfs_HO = np.memmap(savedir+f'/wfs_HO_{tag}.npy', dtype='float32', mode='w+', \
                                            shape=(size, *env.wfs.cam.frame.shape))

frame = 0
nModes = 30

start = time.time()

for j in range(size):

    # choose_spectrum = np.random.choice([0,1,2])  
    
    env.tel.resetOPD()

    env.atm.generateNewPhaseScreen(np.random.randint(0, 4294967296 - 2))
    opd = env.tel.OPD.copy()


    # command = env.M2C_CL@coefficients

    # env.dm.coefs = command.copy()

    # env.tel*env.dm
    env.tel*env.wfs

    wfs_HO[frame] = np.float32(env.wfs.cam.frame.copy())
    # dm_commands[frame] = np.float32(command.copy())

    modal_coefs = np.matmul(opd2m, opd[xpupil, ypupil])

    opd_fit = np.zeros_like(opd)
    opd_fit[xpupil, ypupil] = m2opd[:, :nModes]@modal_coefs[:nModes]

    env.tel.OPD = opd_fit

    env.tel*env.wfs

    wfs_LO[frame] = np.float32(env.wfs.cam.frame.copy())

    frame += 1

    if frame % 1000 == 0:
        print(f"Generated {frame} samples in {time.time()-start} seconds")
        

        with open("making_dataset.txt", "a") as f:
            f.write(f"Generated {frame} samples in {time.time()-start} seconds \n")

        start = time.time()

# dm_commands.flush()
# wfs_frames.flush()
wfs_HO.flush()
wfs_LO.flush()

