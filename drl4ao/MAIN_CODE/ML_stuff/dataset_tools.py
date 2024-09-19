import torch
import numpy as np
import os
from torch.utils.data import DataLoader, Dataset
import yaml
import time


class ImageDataset(Dataset):
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs


    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # Load input image and target from the dataframe
        input_image = self.inputs[idx]
        target_image = self.outputs[idx]
        
        # Convert to float and apply any transformations (like normalization)
        input_image = torch.tensor(input_image, dtype=torch.float32).unsqueeze(0)
        target_image = torch.tensor(target_image, dtype=torch.float32).unsqueeze(0)

        input_image = (input_image - input_image.mean()) / input_image.std()


        return input_image, target_image

class FileDataset(Dataset):
    def __init__(self, input_file_path, target_file_path, split_indices, dm_shape, wfs_shape, scale=1e-6, size=20000, use_mmap=True):
        self.input_file_path = input_file_path
        self.target_file_path = target_file_path
        self.split_indices = split_indices
        self.scale = scale
        self.use_mmap = use_mmap
        self.size = size
        self.dm_shape = dm_shape
        self.wfs_shape = wfs_shape

        if use_mmap:
            self.input_data = np.memmap(self.input_file_path, dtype='float32', mode='r', shape=(self.size, *self.wfs_shape))[self.split_indices]
            self.target_data = np.memmap(self.target_file_path, dtype='float32', mode='r',shape=(self.size, *self.dm_shape) )[self.split_indices]
        else:
            self.input_data = np.load(self.input_file_path)[self.split_indices]
            self.target_data = np.load(self.target_file_path)[self.split_indices]

    def __len__(self):
        return len(self.split_indices)

    def __getitem__(self, idx):
        # Load input image and target from the dataframe
        input_image = self.input_data[idx]
        target_image = self.target_data[idx]
        
        # Convert to float and apply any transformations (like normalization)
        # input_image = torch.tensor(input_image, dtype=torch.float32).unsqueeze(0)
        target_image = torch.tensor(np.arcsinh(target_image / self.scale), dtype=torch.float32).unsqueeze(0)

        # input_image = (input_image - input_image.mean()) / input_image.std()

        #For pyramid
        input_image = torch.tensor(input_image, dtype=torch.float32)
        # input_image = (input_image - input_image.mean()) / input_image.std()
        reshaped_input = input_image.view(2, 24, 2, 24).permute(0, 2, 1, 3).contiguous().view(4, 24, 24)

        return reshaped_input, target_image


def read_yaml_file(file_path):
    with open(file_path, 'r') as file:
        conf = yaml.safe_load(file)
    return conf

def find_main_directory(path):
    # Get the absolute path of the starting point
    path = os.path.abspath(path)
    
    # Traverse up the directory tree
    while path != os.path.dirname(path):  # Check if we reached the root of the filesystem
        if os.path.basename(path) == "RLAO":  # Check if the current directory is named "main"
            return path
        path = os.path.dirname(path)  # Move up one directory
    return None  # Return None if the "main" directory is not found


def make_diverse_dataset(env, size, num_scale=6, min_scale=1e-9, max_scale=1e-8, savedir='', tag='', to_file=False):
    """Creates a pandas DataFrame with wavefront sensor measurements
    and corresponding mirror shapes, generated from normally distributed
    dm coefficients."""

    if to_file:

        dm_commands = np.memmap(savedir+f'/dm_cmds_{tag}.npy', dtype='float32', mode='w+', \
                                                    shape=(size*num_scale, *env.dm.coefs.shape))
        wfs_frames = np.memmap(savedir+f'/wfs_frames_{tag}.npy', dtype='float32', mode='w+', \
                                                shape=(size*num_scale, *env.wfs.cam.frame.shape))

    else:
        dm_commands = np.zeros((size*num_scale, *env.dm.coefs.shape))
        wfs_frames = np.zeros((size*num_scale, *env.wfs.cam.frame.shape))

    frame = 0

    scaling = np.linspace(min_scale, max_scale, num_scale)

    start = time.time()
    for i in range(num_scale):
        for j in range(size):
            

            env.tel.resetOPD()

            command = np.random.randn(*env.dm.coefs.shape) * scaling[i]

            env.dm.coefs = command.copy()

            env.tel*env.dm
            env.tel*env.wfs

            wfs_frames[frame] = np.float32(env.wfs.cam.frame.copy())
            dm_commands[frame] = np.float32(command.copy())

            frame += 1

            if frame % 1000 == 0:
                print(f"Generated {frame} samples in {time.time()-start} seconds")
                start = time.time()

    if to_file:
        dm_commands.flush()
        wfs_frames.flush()
    return wfs_frames, dm_commands


def data_from_stats(env, pwr_spec, size, savedir='', tag=''):



    dm_commands = np.memmap(savedir+f'/dm_cmds_{tag}.npy', dtype='float32', mode='w+', \
                                                shape=(size*num_scale, *env.dm.coefs.shape))
    wfs_frames = np.memmap(savedir+f'/wfs_frames_{tag}.npy', dtype='float32', mode='w+', \
                                                shape=(size*num_scale, *env.wfs.cam.frame.shape))


    frame = 0

    start = time.time()

    for j in range(size):
        

        env.tel.resetOPD()

        coefficients = np.random.normal(0, np.sqr(pwr_spec))

        command = env.M2C_CL@coefficients

        env.dm.coefs = command.copy()

        env.tel*env.dm
        env.tel*env.wfs

        wfs_frames[frame] = np.float32(env.wfs.cam.frame.copy())
        dm_commands[frame] = np.float32(command.copy())

        frame += 1

        if frame % 1000 == 0:
            print(f"Generated {frame} samples in {time.time()-start} seconds")
            start = time.time()

    
    dm_commands.flush()
    wfs_frames.flush()

    return dm_commands, wfs_frames


###### Junk Yard ######

# def get_OL_phase_dataset(env, size):
#     """Creates a pandas DataFrame with wavefront sensor measurements
#     and corresponding mirror shapes."""

#     # Create random OPD maps
#     tel_res = env.dm.resolution

#     true_phase = np.zeros((tel_res,tel_res,size))

#     dataset = pd.DataFrame(columns=['wfs', 'dm'])

#     seeds = np.random.randint(1, 10000, size=size)

#     for i in range(size):
#         env.atm.generateNewPhaseScreen(seeds[i])
#         env.tel*env.wfs

#         dataset.loc[i] = {'wfs': np.array(env.wfs.cam.frame.copy()), 'dm': np.array(env.OPD_on_dm())}
#         true_phase[:,:,i] = env.tel.OPD

#         if i % 100 == 0:
#             print(f"Generated {i} open loop samples")


#     return true_phase, dataset

# def get_CL_phase_dataset(env, size, reconstructor):
#     """Creates a pandas DataFrame with wavefront sensor measurements
#     and corresponding mirror shapes, using the closed loop system."""

#     recontructor.eval()

#     # Create random OPD maps

#     tel_res = env.dm.resolution

#     true_phase = np.zeros((tel_res,tel_res,size))

#     dataset = pd.DataFrame(columns=['wfs', 'dm'])

#     seeds = np.random.randint(1, 10000, size=size)

#     for i in range(size):
#         env.atm.generateNewPhaseScreen(seeds[i])
#         env.tel*env.wfs

#         obs = torch.tensor(env.wfs.cam.frame).clone().detach().float().unsqueeze(0).unsqueeze(0)

#         with torch.no_grad(): 
#             action = reconstructor(obs).squeeze(0).squeeze(0) #env.integrator()

#         pred_OPD = OPD_model(action, env.dm.modes, env.dm.resolution, env.xvalid, env.yvalid)

#         residual_phase = env.tel.OPD.copy() - pred_OPD.squeeze().numpy()

#         env.tel.OPD = residual_phase
#         env.tel*env.wfs

#         dataset.loc[i] = {'wfs': np.array(env.wfs.cam.frame.copy()), 'dm': np.array(env.OPD_on_dm())}
#         true_phase[:,:,i] = env.tel.OPD

#         if i % 100 == 0:
#             print(f"Generated {i} closed loop samples")

#     return true_phase, dataset
