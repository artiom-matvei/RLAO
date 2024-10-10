import torch
import numpy as np
import os
from torch.utils.data import Dataset

class DiffusionDataset(Dataset):
    def __init__(self, HO_file_path, LO_file_path, wfs_shape, device, size=400000, use_mmap=True):
        self.HO_file_path = HO_file_path
        self.LO_file_path = LO_file_path
        self.use_mmap = use_mmap
        self.wfs_shape = wfs_shape
        self.size = size
        self.device = device

        if use_mmap:
            self.HO_data = np.memmap(self.HO_file_path, dtype='float32', mode='r', shape=(self.size, *self.wfs_shape))
            self.LO_data = np.memmap(self.LO_file_path, dtype='float32', mode='r',shape=(self.size, *self.wfs_shape) )
        else:
            self.HO_data = np.load(self.HO_file_path)
            self.LO_data = np.load(self.LO_file_path)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):

        HO_image = self.HO_data[idx]
        LO_image = self.LO_data[idx]
        
        HO_image = torch.tensor(HO_image, dtype=torch.float32)
        LO_image = torch.tensor(LO_image, dtype=torch.float32)
        # input_image = (input_image - input_image.mean()) / input_image.std()
        reshaped_HO = HO_image.view(2, 24, 2, 24).permute(0, 2, 1, 3).contiguous().view(4, 24, 24)
        reshaped_LO = LO_image.view(2, 24, 2, 24).permute(0, 2, 1, 3).contiguous().view(4, 24, 24)

        return reshaped_HO.to(self.device), reshaped_LO.to(self.device)