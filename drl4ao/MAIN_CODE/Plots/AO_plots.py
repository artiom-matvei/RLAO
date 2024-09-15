#%%
import os,sys
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../AO_OOPAO')))

from OOPAO.tools.displayTools import displayMap
import time
import numpy as np
from PO4AO.mbrl_funcsRAZOR import get_env
from ML_stuff.dataset_tools import ImageDataset, FileDataset, make_diverse_dataset, read_yaml_file
from ML_stuff.models import Reconstructor, Reconstructor_2
from gifTools import create_gif
from types import SimpleNamespace
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'inferno'
# SimpleNamespace takes a dict and allows the use of
# keys as attributes. ex: args['r0'] -> args.r0
try:
    args = SimpleNamespace(**read_yaml_file('./Conf/razor_config_po4ao.yaml'))
except:
    args = SimpleNamespace(**read_yaml_file('../Conf/razor_config_po4ao.yaml'))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

savedir = os.path.dirname(__file__)

env = get_env(args)

# env.wfs.reference_slopes_maps = env.wfs.signal_2D.copy()

# %%
#------------- Simple Plot of WFS / Slope map / OPDs -------------#

def wfs_image(env, with_atm=True, seed=0):

    if with_atm:
        env.atm.generateNewPhaseScreen(seed=seed)
        env.tel*env.wfs
    else:
        env.tel.resetOPD()
        env.tel*env.wfs

    frame = env.wfs.cam.frame.copy()

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.imshow(frame, cmap='inferno',\
                     vmin=0, vmax=np.percentile(frame, 99.5))
    ax.axis('off')
    plt.show()


def wfs_vector(env):
    frame = env.wfs.cam.frame.copy()
    slopes = env.wfs.signal_2D.copy()

    image_size = frame.shape[0]
    grid_size = slopes.shape[1]

    spot_spacing = image_size // grid_size  # Distance between spots on the grid

    # Generate grid coordinates (X, Y) for the 10x10 grid
    x = np.linspace(spot_spacing // 2, image_size - spot_spacing // 2, grid_size)
    y = np.linspace(spot_spacing // 2, image_size - spot_spacing // 2, grid_size)
    X, Y = np.meshgrid(x, y)

    x_slopes = slopes[:grid_size, :]
    y_slopes = slopes[grid_size:, :]

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.imshow(frame, cmap='inferno',\
                     vmin=0, vmax=np.percentile(frame, 99.5))

    ax.quiver(X, Y, x_slopes, y_slopes, color='white', scale=5e2)
    ax.axis('off')

    plt.gca().invert_yaxis()
    plt.show()




def slope_map(env, with_atm=True, seed=0):

    if with_atm:
        env.atm.generateNewPhaseScreen(seed=seed)
        env.tel*env.wfs
    else:
        env.tel - env.atm
        env.tel.resetOPD()
        env.tel*env.wfs

        # env.tel + env.atm

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    im1 = ax.imshow(env.wfs.signal_2D.copy().T, cmap='inferno')
    fig.colorbar(im1)
    ax.axis('off')
    plt.show()


def opd_map(env, with_atm=True, seed=0): 
    if with_atm:
        env.atm.generateNewPhaseScreen(seed=seed)

    else:
        env.tel.resetOPD()

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.imshow(env.tel.OPD.copy(), cmap='inferno')
    ax.axis('off')
    plt.show()


# Use this to visualize the OPD induced by the DM commands
def OPD_model(dm_cmd, modes, res):

    vec_cmd = dm_cmd[env.xvalid, env.yvalid]
    dm_opd = np.matmul(vec_cmd,modes.transpose(-1,-2)).squeeze(0)

    dm_opd = torch.reshape(dm_opd, (-1,res,res))

    return dm_opd

def linear_reconstructor(env, opd=False, seed=0):
    # env.atm.generateNewPhaseScreen(seed=seed)
    # env.tel*env.wfs

    integrator = np.matmul(env.reconstructor, env.wfs.signal)

    reconstructor = env.vec_to_img(torch.from_numpy(integrator).float())

    if opd:
        reconstructor = OPD_model(reconstructor, env.dm.modes, env.dm.resolution)[0] * env.tel.pupil

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    im1 = ax.imshow(reconstructor, cmap='inferno')
    fig.colorbar(im1)
    ax.axis('off')
    plt.show()

def influence_functions(env):
    res = env.dm.resolution
    nAct = env.dm.nValidAct
    frame = np.sum(np.reshape(env.dm.modes.copy(),(res, res, nAct))**3, axis=2)

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.imshow(frame, cmap='inferno')
    
    ax.axis('off')
    plt.show()


def basis_grid(env, len=4, modal=True):
    if modal:
        env.tel.resetOPD()

        env.dm.coefs = env.M2C_CL[:,:len**2]
        env.tel*env.dm
        displayMap(env.tel.OPD)
    else:
        env.tel.resetOPD()

        env.dm.coefs = np.eye((env.dm.nValidAct))[:,:len**2]
        env.tel*env.dm
        displayMap(env.tel.OPD - env.tel.OPD.mean())

def basis_distribution(env):
    size = 100
    zonal_coefs = np.zeros((size, env.dm.nValidAct))
    modal_coefs = np.zeros((size, env.M2C_CL.shape[1]))

    zonal_modes = env.dm.modes.copy()
    zonal_proj = np.matmul(np.linalg.inv(np.matmul(zonal_modes.T, zonal_modes)), zonal_modes.T)

    zonal_proj /= np.linalg.norm(zonal_proj, axis=1)[:, None]

    env.tel.resetOPD()

    env.dm.coefs = env.M2C_CL
    env.tel*env.dm
    res = env.dm.resolution
    zernike_modes = env.tel.OPD.copy().reshape(res*res, -1)
    zernike_proj = np.matmul(np.linalg.inv(np.matmul(zernike_modes.T, zernike_modes)), zernike_modes.T)

    zernike_proj /= np.linalg.norm(zernike_proj, axis=1)[:, None]

    for i in range(size):
        env.atm.generateNewPhaseScreen(2389744 * i)
        opd = env.tel.OPD.copy()

        zonal_coefs[i] = np.matmul(zonal_proj, opd.reshape(res*res))
        modal_coefs[i] = np.matmul(zernike_proj, opd.reshape(res*res))

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].plot(np.mean(np.square(zonal_coefs), axis=0), color='k')
    ax[0].set_yscale('log')
    ax[0].set_title('Power Spectrum of Atmosphere in Zonal Basis')

    ax[1].plot(np.mean(np.square(modal_coefs), axis=0), color='k')
    ax[1].set_yscale('log')
    ax[1].set_title('Power Spectrum of Atmosphere in Modal Basis')

    plt.show()


# ANIMATE THE IM
def animate_im(env, frame_rate=10):

    slope_cube = np.zeros((env.M2C_CL.shape[1], *env.wfs.signal_2D.shape))
    dm_cube = np.zeros((env.M2C_CL.shape[1], env.nActuator, env.nActuator))
    opd_cube = np.zeros((env.M2C_CL.shape[1], *env.tel.OPD.shape))


    env.wfs.cam.readoutNoise = 0
    env.wfs.cam.photonNoise = False
    env.wfs.cam.darkCurrent = 0
    env.wfs.cam.FWC = None
    env.change_mag(4)

    env.tel - env.atm
    env.tel.resetOPD()

    for i in range(env.M2C_CL.shape[1]):
        env.dm.coefs = env.M2C_CL[:,i]*1e-6
        env.tel*env.dm
        env.tel*env.wfs

        norm_slope = np.linalg.norm(env.wfs.signal_2D)
        norm_dm = np.linalg.norm(env.dm.coefs)
        norm_opd = np.linalg.norm(env.tel.OPD)

        slope_cube[i] = env.wfs.signal_2D.copy()/ norm_slope
        dm_cube[i] = env.vec_to_img(torch.from_numpy(env.dm.coefs).float())/ norm_dm
        opd_cube[i] = env.tel.OPD.copy()/ norm_opd

    create_gif(slope_cube, frame_rate=frame_rate, output_file='slope_map.gif')
    create_gif(dm_cube, frame_rate=frame_rate, output_file='dm_map.gif')
    create_gif(opd_cube, frame_rate=frame_rate, output_file='opd_map.gif')



    return



def fourier_series(n_deg):
    x = np.linspace(0, 2*np.pi, 1000)
    y = np.zeros_like(x)

    for i in range(len(y) - 1):
        y[i+1] = y[i] + np.random.randn(1) + 0.01

    fft_coeffs = np.fft.fft(y)

# Get the corresponding frequencies
    frequencies = np.fft.fftfreq(len(y), x[1] - x[0])

    # Reconstruct the signal using a limited number of Fourier components
    n_terms = 10  # Number of Fourier terms to include in the reconstruction

    # Initialize the reconstructed signal as zero
    y_reconstructed = np.zeros_like(y)

    # Reconstruct the signal using n_terms Fourier components
    for k in range(n_deg):
        # Use both positive and negative frequencies (symmetry of FFT)
        y_reconstructed += (fft_coeffs[k] * np.exp(2j * np.pi * frequencies[k] * x)).real
        y_reconstructed += (fft_coeffs[-k-1] * np.exp(2j * np.pi * frequencies[-k-1] * x)).real

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(x, y, label='Original Signal', color='k', lw=3)
    ax.plot(x, y_reconstructed/1000, label=f'{n_deg} Degrees of Freedom', linestyle='--', color='r', lw=3)
    ax.set_xlim([1, 6])
    ax.legend()

    plt.show()


# %%
# Visualize how a shackhartmann creates a slope map using 1D functions
def shwfs_1D():

    def f(x):
        return np.sin(10*x)/2 + np.exp(x)

    x = np.linspace(-1, 1, 1000)
    y = f(x)

    sample_points = np.linspace(-1, 1, 10)
    sample_values = f(sample_points)
    midpoints = (sample_points[:-1] + sample_points[1:]) / 2
    midpoint_values = f(midpoints)

    # Compute the derivative using finite differences
    dx = sample_points[1] - sample_points[0]
    dy_dx = (sample_values[1:] - sample_values[:-1])/ dx

    # Initialize the approximation array
    approx_y = np.zeros_like(x)

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    for i in range(len(sample_points) - 1):
    # Get the slope (dy/dx) and intercept for each linear segment
        slope = dy_dx[i]
        intercept = midpoint_values[i] - slope * midpoints[i]
        
        # Create the linear function between sample points
        mask = (x >= sample_points[i]) & (x < sample_points[i+1])
        approx_y[mask] = slope * x[mask] + intercept

        ax.plot(x[mask], approx_y[mask], label='Linear Approximation', linestyle='-', color='r', lw=3)

    # For the last segment
    # approx_y[x >= sample_points[-2]] = dy_dx[-1] * x[x >= sample_points[-2]] + (midpoint_values[-2] - dy_dx[-1] * sample_points[-2])
    ax.plot(x, y, color='k', lw=1)
    for i in range(len(sample_points)):
        ax.axvline(sample_points[i], color='k', linestyle='--', lw=1, alpha=0.5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    # ax.axis('off')
    plt.show()

# %%
def wf_recon_test(env, reconstructor):


    wfsf, dmc = make_diverse_dataset(env, size=1, num_scale=3,\
                        min_scale=1e-6, max_scale=1e-6)

    reconstructor.eval()
    # Make predictions
    obs = torch.tensor(wfsf).float().unsqueeze(1)

    with torch.no_grad():
        pred = reconstructor(obs)
        # pred_OPD = OPD_model(network(obs), modes, res)


    # Run ground truth commands through the model

    # Reshape commands into image
    cmd_img = np.array([env.vec_to_img(torch.tensor(i).float()) for i in dmc])

    # gt_opd = OPD_model(torch.tensor(cmd_img).unsqueeze(1), modes, res)


    fig, ax = plt.subplots(3,3, figsize=(10,10))

    vrange = 3

    for i in range(3):
        # cax1 = ax[0,i].imshow(wfsf[i])
        cax2 = ax[0,i].imshow(pred[i].squeeze(0).detach().numpy(), vmin=-vrange, vmax=vrange)
        cax3 = ax[1,i].imshow(np.arcsinh(cmd_img[i] / 1e-6), vmin=-vrange, vmax=vrange)
        cax4 = ax[2,i].imshow(np.arcsinh(cmd_img[i] / 1e-6) - pred[i].squeeze(0).detach().numpy(),\
                                                                    vmin=-vrange, vmax=vrange)


        ax[0,i].axis('off')
        ax[1,i].axis('off')
        ax[2,i].axis('off')
        # ax[3,i].axis('off')

        # ax[0,i].set_title('Input WFS Image', size=15)
        ax[0,i].set_title('Reconstructed Phase', size=15)
        ax[1,i].set_title('Ground Truth Phase', size=15)
        ax[2,i].set_title('Difference', size=15)

    # plt.tight_layout()

    plt.show()

# %%
