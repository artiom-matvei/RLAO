#%%
import torch
import numpy as np
import os, sys
import matplotlib.pyplot as plt
from score_models import ScoreModel, NCSNpp

# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

script_dir = os.path.dirname(os.path.abspath(__file__))
# Print the size in mb of the dataset file
lr = np.load(f'{script_dir}/images/lr.npy')
hr = np.load(f'{script_dir}/images/hr.npy')

model = ScoreModel(checkpoints_directory=f'{script_dir}/datasets/cp_unconditional/', device=device)

B = 1024
channels = 4


# %%
num_steps = 100
dt = -1/num_steps #reverse time

s_min = model.sde.sigma_min
s_max = model.sde.sigma_max

with torch.no_grad():
    for eta in np.logspace(1, 2.2, 10):
    # eta = 0.05

        y = torch.from_numpy(lr[:B]).to(device)

        x_t = torch.normal(0, s_max, (B, channels, 24, 24)).to(device)

        t_start = 1 # if not start_from_y else np.log(np.std(lr[0][0] - hr[0][0])) / np.log(s_max / s_min)

        for i, t in enumerate(np.linspace(t_start, 0, num_steps)):
            print(f'Step {i}/{num_steps}')
            t = torch.tensor(t).to(device) * torch.ones(B).to(device)
            z = torch.randn_like(x_t).to(device)
            dw = abs(dt)**(1/2) * z
            g = model.sde.diffusion(t, x_t)

            sig_t = model.sde.sigma(t).unsqueeze(1).unsqueeze(2).unsqueeze(3)

            score_likelihood = (y - x_t) / (sig_t ** 2 + eta**2)
            score_prior = model.score(t, x_t)

            
            dx = - g**2 * ( score_likelihood + score_prior) * dt + g * dw

            x_t += dx



        lr_fft2 = np.fft.fft2(lr.sum(axis=1))
        lr_fft_shifted = np.fft.fftshift(lr_fft2, axes=(-2, -1))

        hr_fft2 = np.fft.fft2(hr.sum(axis=1))
        hr_fft_shifted = np.fft.fftshift(hr_fft2, axes=(-2, -1))


        sam_fft2 = np.fft.fft2(x_t.detach().cpu().sum(dim=1))
        sam_fft_shifted = np.fft.fftshift(sam_fft2, axes=(-2, -1))


        def distance_matrix(n, c_row, c_col):
                # Create a grid of indices
                i, j = np.indices((n, n))
                
                # Calculate the distance for each pixel
                distances = np.sqrt((i - c_row)**2 + (j - c_col)**2)
                
                return distances

        n = 24

        i_c, j_c = n // 2, n // 2

        r = distance_matrix(n, i_c, j_c)

        max_radius = n // 2
        mesh = np.linspace(0, max_radius, max_radius*2)

        # Compute power spectrum for each batch
        batch_pow_lr = []
        batch_pow_hr = []
        batch_pow_sam = []

        for b in range(B):
            pow_lr = []
            pow_hr = []
            pow_sam = []
            for i in range(len(mesh) - 1):
                # Create a mask for the current radial bin
                mask = (mesh[i] <= r) & (r < mesh[i + 1])
                if np.any(mask):  # Avoid issues with empty bins
                    pow_lr.append(np.mean(np.abs(lr_fft_shifted[b][mask])))
                    pow_hr.append(np.mean(np.abs(hr_fft_shifted[b][mask])))
                    pow_sam.append(np.mean(np.abs(sam_fft_shifted[b][mask])))
                else:
                    pow_lr.append(0)  # Handle empty bins gracefully
                    pow_hr.append(0)
                    pow_sam.append(0)
            batch_pow_lr.append(pow_lr)
            batch_pow_hr.append(pow_hr)
            batch_pow_sam.append(pow_sam)

        # Convert results to a numpy array for further analysis
        batch_pow_lr = np.array(batch_pow_lr)
        batch_pow_hr = np.array(batch_pow_hr)
        batch_pow_sam = np.array(batch_pow_sam)

        # plt.plot(np.mean(batch_pow_lr, axis=0), label="LR")
        # plt.plot(np.mean(batch_pow_hr, axis=0), label="HR")
        # plt.plot(np.mean(batch_pow_sam, axis=0), label="Sampled")
        # plt.yscale('log')
        # plt.legend()

        # plt.show()
        np.save(f'{script_dir}/images/batch_pow_lr_{eta:.0f}.npy', batch_pow_lr)
        np.save(f'{script_dir}/images/batch_pow_hr_{eta:.0f}.npy', batch_pow_hr)
        np.save(f'{script_dir}/images/batch_pow_sam_{eta:.0f}.npy', batch_pow_sam)

        

# torch.save(x_t, f'{script_dir}/images/samples_unc.pt')
# %%
