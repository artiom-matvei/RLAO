import numpy as np
import torch
from score_models import ScoreModel
from tqdm import tqdm
import os

# === Settings ===
batch_size = 1000
base_dir = os.path.dirname(os.path.abspath(__file__))
input_file_path = base_dir + "/../wf_recon/datsets/thesis_data/wfs_frames_filtered.npy"
output_file_path = base_dir + "/../wf_recon/datsets/thesis_data/wfs_frames_augmented.npy"
dataset_size = 400_000
wfs_shape = (48, 48)
reshaped_shape = (4, 24, 24)
steps = 1000

# === Load model ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ScoreModel(checkpoints_directory= base_dir + '/datasets/checkpoints2/', device=device)
model.eval()

# === Prepare input and output memmaps ===
input_data = np.memmap(input_file_path, dtype='float32', mode='r', shape=(dataset_size, *wfs_shape))
output_data = np.memmap(output_file_path, dtype='float32', mode='w+', shape=(dataset_size, *wfs_shape))

# === Batch processing ===
num_batches = (dataset_size + batch_size - 1) // batch_size

for i in tqdm(range(num_batches), desc="Sampling with diffusion"):
    start = i * batch_size
    end = min((i + 1) * batch_size, dataset_size)
    batch_inputs = input_data[start:end]  # shape: [B, 48, 48]

    # Convert and reshape for diffusion model
    batch_inputs_tensor = torch.from_numpy(batch_inputs).to(torch.float32).to(device)  # [B, 48, 48]
    
    # Reshape: [B, 48, 48] -> [B, 4, 24, 24]
    reshaped = batch_inputs_tensor.view(-1, 2, 24, 2, 24).permute(0, 1, 3, 2, 4).contiguous().view(-1, 4, 24, 24)

    # Sampling
    with torch.no_grad():
        samples = model.sample(
            shape=[reshaped.shape[0], *reshaped_shape],
            steps=steps,
            condition=(reshaped,)
        )

    # Reshape back to [B, 48, 48]
    recovered = samples.view(-1, 2, 2, 24, 24).permute(0, 1, 3, 2, 4).contiguous().view(-1, 48, 48)

    # Save to memmap
    output_data[start:end] = recovered.cpu().numpy()

    with open(base_dir + "/augmenting.txt", "a") as f:  # 'a' mode appends to the file
        f.write(f"Sampled {end} samples \n")

# Flush to disk
output_data.flush()
