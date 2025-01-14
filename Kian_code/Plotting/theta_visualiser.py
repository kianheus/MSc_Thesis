# Recreate the design with a dynamic background and immersive grid
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import matplotlib.pyplot as plt
import Learning.training_data as training_data
import Double_Pendulum.Lumped_Mass.transforms as transforms
from torch import Tensor
import torch
from matplotlib.animation import FuncAnimation, ArtistAnimation



def x_func(rp: dict, q: Tensor) -> Tensor:
    
    # x position of the end effector
    
    x = rp["l1"] * torch.cos(q[0]) + rp["l2"] * torch.cos(q[1])
    
    return x
    
    
def y_func(rp: dict, q: Tensor) -> Tensor:
    
    # y position of the end effector
    
    y = rp["l1"] * torch.sin(q[0]) + rp["l2"] * torch.sin(q[1])
    
    return y




class theta_plotter:

    def __init__(self, rp, n_lines, mapping_functions = (x_func, y_func)):

        # Create n lines of points across q1 and q2 across the training range for visualization
        q1 = torch.linspace(training_data.q1_low, training_data.q1_high, n_lines)
        q2 = torch.linspace(training_data.q2_low, training_data.q2_high, n_lines)

        # Turn these lines into meshgrids
        self.q1_grid, self.q2_grid = torch.meshgrid(q1, q2, indexing="ij")

        # Flatten for easier computation with vmap
        q_combined = torch.stack((self.q1_grid.flatten(), self.q2_grid.flatten()), dim=-1)

        # Calculate coordinate change
        Theta1 = torch.vmap(mapping_functions[0], in_dims=(None, 0))(rp, q_combined)
        Theta2 = torch.vmap(mapping_functions[1], in_dims=(None, 0))(rp, q_combined)

        self.Theta1 = Theta1.view(n_lines, n_lines)
        self.Theta2 = Theta2.view(n_lines, n_lines)

    def make_animation(self, file_name, duration, fps, stride):
        
        """
        Takes file_name (including filetype) and generates an animation which shows how
        orthogonal lines move under the coordinate change from q to theta.         
        """

        interval = int(1000/fps)
        frames = int(1000*duration/interval)
        output_path = "Plotting/Plots/" + file_name

        # Create the figure with corrected settings
        fig, ax = plt.subplots(figsize=(10, 10), dpi=400)

        # Store frames for ArtistAnimation
        frames_artists = []

        q1_grid_thin = self.q1_grid[::stride, ::stride] 
        q2_grid_thin = self.q2_grid[::stride, ::stride] 
        Theta1_thin = self.Theta1[::stride, ::stride]
        Theta2_thin = self.Theta2[::stride, ::stride]

        # Loop over all frames and morph from original coordinates q to new coordinates theta
        # Append artists of each frame to be animated later
        for frame in range(frames):
            t = frame / (frames - 1)  # Normalized time from 0 to 1
            plot_Theta1 = (1 - t) * q1_grid_thin + t * Theta1_thin
            plot_Theta2 = (1 - t) * q2_grid_thin + t * Theta2_thin

            artists = []  # Collect artists for this frame
            for i in range(Theta1_thin.shape[1]):
                h_line, = ax.plot(plot_Theta1[i, :], plot_Theta2[i, :], color='black', lw=0.3)
                v_line, = ax.plot(plot_Theta1[:, i], plot_Theta2[:, i], color='black', lw=0.3)
                artists.extend([h_line, v_line])

            frames_artists.append(artists)

        # Duplicate the final frame another second to improve viewability
        final_artists = frames_artists[-1]
        for _ in range(fps):
            frames_artists.append(final_artists)    

        # Create the animation
        ani = ArtistAnimation(fig, frames_artists, interval=interval, blit=True)

        # Save the animation as MP4
        ani.save(output_path, writer="ffmpeg", dpi=400, bitrate=-1)
        plt.close(fig)

    def make_figure(self, file_name):

        # Set the output path
        output_path = "Plotting/Plots/" + file_name

        # Create the figure with corrected settings
        fig, ax = plt.subplots(figsize=(10, 10), dpi=400)

        # Plot the transformed grid lines
        for i in range(self.Theta1.shape[1]):  # Draw fewer lines for clarity
            ax.plot(self.Theta1[i, :], self.Theta2[i, :], color='black', lw=0.3, alpha=0.9)
            ax.plot(self.Theta1[:, i], self.Theta2[:, i], color='black', lw=0.3, alpha=0.9)

        # Save the figure
        plt.savefig(output_path, dpi=400, bbox_inches='tight', pad_inches=0)
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