
#%%
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from score_models import ScoreModel, NCSNpp

from data_loading.dataset_tools import DiffusionDataset

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