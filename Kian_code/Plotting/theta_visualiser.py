# Recreate the design with a dynamic background and immersive grid
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import matplotlib.pyplot as plt
import Learning.training_data as training_data
import Double_Pendulum.Lumped_Mass.transforms as transforms
from torch import Tensor
import torch
from matplotlib.animation import FuncAnimation, ArtistAnimation


def test_theta_1(rp: dict, q: Tensor) -> Tensor:
    
    h1 = q[0]*q[0]
    
    return h1
    
    
def test_theta_2(rp: dict, q: Tensor) -> Tensor:
    
    h2 = q[1]
    
    return h2

def plotter(rp):
    # Generate the curved lines to simulate a coordinate transformation effect
    n_lines = 51
    frames = 37  # Number of frames for the animation
    q1 = torch.linspace(training_data.q1_low, training_data.q1_high, n_lines)
    q2 = torch.linspace(training_data.q2_low, training_data.q2_high, n_lines)

    # Create the meshgrid for the grid of points
    q1_grid, q2_grid = torch.meshgrid(q1, q2, indexing="ij")  # Shape: (101, 101)

    # Flatten for easier computation with vmap
    q_combined = torch.stack((q1_grid.flatten(), q2_grid.flatten()), dim=-1)  # Shape: (n_lines * n_lines, 2)

    Theta1 = torch.vmap(transforms.analytic_theta_1, in_dims=(None, 0))(rp, q_combined)
    Theta2 = torch.vmap(transforms.analytic_theta_2, in_dims=(None, 0))(rp, q_combined)

    Theta1 = Theta1.view(n_lines, n_lines)
    Theta2 = Theta2.view(n_lines, n_lines)

    # Recreate the figure with corrected settings
    fig, ax = plt.subplots()

    output_path = "Plotting/Plots/morphing_animation.mp4"

    # Store frames for ArtistAnimation
    frames_artists = []

    for frame in range(frames):
        t = frame / (frames - 1)  # Normalized time from 0 to 1
        plot_Theta1 = (1 - t) * q1_grid + t * Theta1
        plot_Theta2 = (1 - t) * q2_grid + t * Theta2

        artists = []  # Collect artists for this frame
        for i in range(Theta1.shape[1]):
            h_line, = ax.plot(plot_Theta1[i, :], plot_Theta2[i, :], color='black', lw=0.2, alpha=0.9)
            v_line, = ax.plot(plot_Theta1[:, i], plot_Theta2[:, i], color='black', lw=0.2, alpha=0.9)
            artists.extend([h_line, v_line])

        frames_artists.append(artists)

    # Create the animation
    ani = ArtistAnimation(fig, frames_artists, interval=20, blit=True)

    # Save the animation as MP4
    ani.save(output_path, writer="ffmpeg")#, dpi=400, bitrate=1800)
    plt.close(fig)


    """
    # Define a dark blue/purple background color
    colors = [(0/255, 10/255, 80/255), (60/255, 0, 65/255)]  # Dark blue to purple
    background_cmap = LinearSegmentedColormap.from_list("custom_gradient", colors)

    # Generate a gradient background
    bg_x = np.linspace(0, 1, 400)
    bg_y = np.linspace(0, 1, 300)
    bg_X, bg_Y = np.meshgrid(bg_x, bg_y)
    bg_Z = np.sin(bg_X * np.pi) * np.cos(bg_Y * np.pi)

    # Create the background as an image
    ax.imshow(bg_Z, cmap=background_cmap, origin='lower', extent=[training_data.q1_low, training_data.q1_high, 
                                                                training_data.q2_low, training_data.q2_high], 
                                                                alpha=0.95)
    """