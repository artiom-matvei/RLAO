
#%%
import torch
import numpy as np
import os, sys
import matplotlib.pyplot as plt
from score_models import ScoreModel, NCSNpp

from data_loading.dataset_tools import DiffusionDataset



sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from PO4AO.mbrl import get_env
from ML_stuff.dataset_tools import read_yaml_file
from types import SimpleNamespace
try:
    args = SimpleNamespace(**read_yaml_file('./Conf/papyrus_config.yaml'))
except:
    args = SimpleNamespace(**read_yaml_file('../Conf/papyrus_config.yaml'))


env = get_env(args)
#%%

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

HO_file_path = '/home/parker09/projects/def-lplevass/parker09/RLAO/drl4ao/MAIN_CODE/diffusion/datasets/wfs_HO_diff.npy'
LO_file_path = '/home/parker09/projects/def-lplevass/parker09/RLAO/drl4ao/MAIN_CODE/diffusion/datasets/wfs_LO_diff.npy'
wfs_shape = (48, 48)

dataset = DiffusionDataset(HO_file_path, LO_file_path, wfs_shape, device, size=400000, use_mmap=True)

net = NCSNpp(channels=4, nf=64, ch_mult=(2, 2, 2), condition=("input",), condition_input_channels=4)
sbm = ScoreModel(net, sigma_min=1e-3, sigma_max=1100)

# hi, lo = dataset.__getitem__(19204)

# frame = torch.zeros((24*4, 24*2))
# for i in range(4):
#     frame[24*i: 24*i + 24, :] = torch.cat([hi[i],lo[i]], dim=1)


# plt.imshow(np.log10(frame + 1))
# plt.title('Raw ATM     50 Modes')
# plt.axis('off')
# plt.show()

# %%
checkpoint_dir = '/home/parker09/projects/def-lplevass/parker09/RLAO/drl4ao/MAIN_CODE/diffusion/datasets/checkpoints2'

sbm.fit(dataset, learning_rate=1e-4, epochs=100000, batch_size=16,\
        checkpoints=10, checkpoints_directory=checkpoint_dir,\
        models_to_keep=2)

infer = False

if infer:
        # Get the env somehow
        model = ScoreModel(checkpoints_directory='//Users/parkerlevesque/School/Research/AO/RLAO/drl4ao/MAIN_CODE/diffusion/checkpoints2/checkpoints2')
        M2OPD = np.load('/Users/parkerlevesque/School/Research/AO/RLAO/drl4ao/MAIN_CODE/predictiveControl/saved_filters/M2OPD_300modes.npy')
        OPD2M = np.linalg.pinv(M2OPD)
        xpupil, ypupil = np.where(env.tel.pupil == 1)
        env.tel.resetOPD()


        env.atm.generateNewPhaseScreen(np.random.randint(0, 4294967296 - 2))
        opd = env.tel.OPD.copy()


        env.tel*env.wfs

        wfs_HO = np.float32(env.wfs.cam.frame.copy())
        # dm_commands[frame] = np.float32(command.copy())

        modal_coefs = np.matmul(OPD2M, opd[xpupil, ypupil])

        opd_fit = np.zeros_like(opd)
        opd_fit[xpupil, ypupil] = M2OPD[:, :50]@modal_coefs[:50]

        env.tel.OPD = opd_fit

        env.tel*env.wfs

        wfs_LO = np.float32(env.wfs.cam.frame.copy())

        HO_image = torch.tensor(wfs_HO, dtype=torch.float32)
        LO_image = torch.tensor(wfs_LO, dtype=torch.float32)
        # input_image = (input_image - input_image.mean()) / input_image.std()
        reshaped_HO = HO_image.view(2, 24, 2, 24).permute(0, 2, 1, 3).contiguous().view(4, 24, 24)
        reshaped_LO = LO_image.view(2, 24, 2, 24).permute(0, 2, 1, 3).contiguous().view(4, 24, 24)

        samples = model.sample(shape=[10, *reshaped_HO.shape], steps=1000, condition=(reshaped_LO.unsqueeze(0).repeat(10,1,1,1),))

        fig,ax = plt.subplots(10,3, figsize=(10, 30))

        for j in range(10):
                ax[j,0].imshow(reshaped_LO.sum(dim=0))
                ax[j,0].set_title('Low Res WFS')
                ax[j,1].imshow(reshaped_HO.sum(dim=0))
                ax[j,1].set_title('High Res wfs')
                ax[j,2].imshow(samples[j].sum(dim=0))
                ax[j,2].set_title('Sample from Model')

        for i in range(3):
                for j in range(10):
                        ax[j,i].axis('off')


        size = 100

        hr = torch.zeros((size, 4, 24, 24))
        lr = torch.zeros((size, 4, 24, 24))

        for i in range(size):

                env.atm.generateNewPhaseScreen(np.random.randint(0, 4294967296 - 2))
                opd = env.tel.OPD.copy()

                env.tel*env.wfs

                wfs_HO = np.float32(env.wfs.cam.frame.copy())
                # dm_commands[frame] = np.float32(command.copy())

                modal_coefs = np.matmul(OPD2M, opd[xpupil, ypupil])

                opd_fit = np.zeros_like(opd)
                opd_fit[xpupil, ypupil] = M2OPD[:, :50]@modal_coefs[:50]

                env.tel.OPD = opd_fit

                env.tel*env.wfs

                wfs_LO = np.float32(env.wfs.cam.frame.copy())

                HO_image = torch.tensor(wfs_HO, dtype=torch.float32)
                LO_image = torch.tensor(wfs_LO, dtype=torch.float32)
                # input_image = (input_image - input_image.mean()) / input_image.std()
                reshaped_HO = HO_image.view(2, 24, 2, 24).permute(0, 2, 1, 3).contiguous().view(4, 24, 24)
                reshaped_LO = LO_image.view(2, 24, 2, 24).permute(0, 2, 1, 3).contiguous().view(4, 24, 24)

                hr[i] = reshaped_HO
                lr[i] = reshaped_LO


        samples = model.sample(shape=[size, *reshaped_HO.shape], steps=1000, condition=(lr,))





        lr_fft2 = np.fft.fft2(lr.sum(dim=1))
        lr_fft_shifted = np.fft.fftshift(lr_fft2)

        hr_fft2 = np.fft.fft2(hr.sum(dim=1))
        hr_fft_shifted = np.fft.fftshift(hr_fft2)


        sam_fft2 = np.fft.fft2(samples.sum(dim=1))
        sam_fft_shifted = np.fft.fftshift(sam_fft2)


        def distance_matrix(n, c_row, c_col):
                # Create a grid of indices
                i, j = np.indices((n, n))
                
                # Calculate the distance for each pixel
                distances = np.sqrt((i - c_row)**2 + (j - c_col)**2)
                
                return distances

        i_c, j_c = 12,12

        r = distance_matrix(24, i_c, j_c)

        max_radius = 24
        mesh = np.linspace(0, max_radius, max_radius*2)

        pow = [np.mean(np.abs(lr_fft_shifted[(mesh[i] <= r)&(r < mesh[i+1])])) for i in range(len(mesh)-1)]
