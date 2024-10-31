import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation,FFMpegWriter, PillowWriter

# Function to create a single frame of the image
def create_frame(data_slice):
    fig, ax = plt.subplots()
    cmap = plt.cm.inferno
    norm = mcolors.Normalize(vmin=np.nanmin(data_slice), vmax=np.nanmax(data_slice))
    
    # Create a masked array for NaN values
    masked_data = np.ma.masked_invalid(data_slice)
    
    # Set up the color map to handle NaNs
    cmap.set_bad(color='white')
    
    cax = ax.imshow(masked_data, cmap=cmap, norm=norm)
    fig.colorbar(cax)
    plt.close(fig)  # Close the figure to avoid displaying it in an interactive environment
    return fig

# Function to update the frame
def update_frame(frame, data_cube, img, ax=None, title=False):
    data_slice = data_cube[frame]
    masked_data = np.ma.masked_invalid(data_slice)
    img.set_data(masked_data)
    if title:
        ax.set_title(f"Frame {frame + 1}")
    return img,

# Function to create the GIF
def create_gif(data_cube, frame_rate, output_file, vmin = None, vmax = None, dpi=200, title=False):
    num_frames = data_cube.shape[0]
    fig, ax = plt.subplots()
    
    # Create the first frame
    data_slice = data_cube[0]
    cmap = plt.cm.inferno
    cmap.set_bad(color=(0, 0, 0, 0))
    masked_data = np.ma.masked_invalid(data_slice)
    if vmin is None:
        vmin=np.nanmin(data_cube)
    if vmax is None:
        vmax=np.nanmax(data_cube)
    img = ax.imshow(masked_data, cmap=cmap, 
                    vmin=vmin, 
                    vmax=vmax)
    ax.axis('off')
    
    # Animation function
    ani = FuncAnimation(fig, update_frame, 
                        frames=num_frames, 
                        fargs=(data_cube, img), 
                        interval=1000/frame_rate)
    
    # Save the animation as an MP4
    # writer = FFMpegWriter(fps=frame_rate)
    writer = PillowWriter(fps=frame_rate)
    ani.save(output_file, writer=writer, dpi=dpi)
    plt.close(fig)

# Example usage
if __name__ == "__main__":
    # Create some example data
    data_cube = np.random.rand(10, 100, 100)
    data_cube[0, 0, 0] = np.nan  # Introduce a NaN value
    
    create_gif(data_cube, frame_rate=2, output_file="output.gif")
