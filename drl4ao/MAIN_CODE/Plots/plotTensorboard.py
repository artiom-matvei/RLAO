#%%
import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np

plt.style.use('ggplot')

def extract_scalars_from_events(logdir):
    """Extracts scalar data from PyTorch TensorBoard event files in a directory."""
    scalar_data = {}

    for subdir in sorted(os.listdir(logdir)):  # Iterate over subdirectories (experiments)
        subdir_path = os.path.join(logdir, subdir)
        if not os.path.isdir(subdir_path):
            continue  # Skip non-directories

        for file in os.listdir(subdir_path):
            if "events.out.tfevents" in file:
                filepath = os.path.join(subdir_path, file)

                event_acc = EventAccumulator(filepath)
                event_acc.Reload()  # Load the event file

                for tag in event_acc.Tags()["scalars"]:  # Iterate over scalar tags (e.g., loss, accuracy)
                    exp_name = subdir  # Use subdir as experiment label
                    
                    if exp_name not in scalar_data:
                        scalar_data[exp_name] = {}
                    if tag not in scalar_data[exp_name]:
                        scalar_data[exp_name][tag] = {'steps': [], 'values': []}

                    for scalar_event in event_acc.Scalars(tag):
                        scalar_data[exp_name][tag]['steps'].append(scalar_event.step)
                        scalar_data[exp_name][tag]['values'].append(scalar_event.value)

    return scalar_data

def plot_tensorboard_scalars(logdir):
    """Plots scalars from TensorBoard event files (using PyTorch & tensorboard)."""
    scalar_data = extract_scalars_from_events(logdir)

    if not scalar_data:
        print("No scalar data found.")
        return

    # Dictionary to store values across multiple runs at each step
    aggregated_data = {}

    for exp_name, tags in scalar_data.items():
        if "CL_OOPAO" not in exp_name and "m2m" not in exp_name:
            continue  # Only process experiments containing "Multi"
        
        for tag, data in tags.items():
            if "return" not in tag:
                continue  # Only process scalars containing "return"

            for step, value in zip(data['steps'], data['values']):
                if step not in aggregated_data:
                    aggregated_data[step] = []
                aggregated_data[step].append(value)

   # Convert to sorted lists for plotting
    steps = sorted(aggregated_data.keys())
    means = np.array([np.mean(aggregated_data[step]) for step in steps])
    maxs = np.array([np.max(aggregated_data[step]) for step in steps])
    mins = np.array([np.min(aggregated_data[step]) for step in steps])

    # Plot mean with shaded error region
    plt.figure(figsize=(10, 6))
    plt.plot(steps, means, label="Mean Return", color='b')
    plt.fill_between(steps, mins, maxs, alpha=0.3, color='b', label="Min / Max")

    plt.xlabel("Steps")
    plt.ylabel("Return")
    plt.title("Mean Episodic Return")
    plt.legend()
    plt.show()


# Example usage
log_directory = "/home/parker09/projects/def-lplevass/parker09/RLAO/drl4ao/MAIN_CODE/runs"
plot_tensorboard_scalars(log_directory)

# %%
