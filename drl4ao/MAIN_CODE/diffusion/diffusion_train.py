import torch
import numpy as np
import os
from score_models import ScoreModel, NCSNpp

from data_loading.dataset_tools import DiffusionDataset

HO_file_path = 'datasets/wfs_HO_diff.npy'
LO_file_path = 'datasets/wfs_LO_diff.npy'
wfs_shape = (48, 48)

dataset = DiffusionDataset(HO_file_path, LO_file_path, wfs_shape, size=400000, use_mmap=True)

net = NCSNpp(channels=4, nf=64, ch_mult=(2, 2, 2), conditions=("input_tensor",), condition_channels=(4,))
sbm = ScoreModel(net, sigma_min=1e-3, sigma_max=1100)