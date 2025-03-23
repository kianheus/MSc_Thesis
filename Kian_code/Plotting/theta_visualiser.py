# Recreate the design with a dynamic background and immersive grid
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import matplotlib.pyplot as plt
import Learning.training_data as training_data
import Double_Pendulum.transforms as transforms
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

    def __init__(self, rp, device, n_lines, mapping_functions = (x_func, y_func), mask_split = [-torch.pi/2]):

        # Create n lines of points across q1 and q2 across the training range for visualization
        q1 = torch.linspace(training_data.q1_low, training_data.q1_high, n_lines)
        q2 = torch.linspace(training_data.q2_low, training_data.q2_high, n_lines)

        # Turn these lines into meshgrids
        self.q1_grid, self.q2_grid = torch.meshgrid(q1, q2, indexing="ij")

        # Apply the slicing condition: keep points where q2 is in [q1, q1 + π]
        #self.valid_mask = (self.q2_grid >= self.q1_grid) & (self.q2_grid <= self.q1_grid + torch.pi)

        self.colors = ["k"]

        self.valid_mask = ((self.q2_grid >= self.q1_grid) & 
                        (self.q2_grid <= self.q1_grid + torch.pi) &
                        (self.q1_grid >= mask_split[0]) & 
                        (self.q1_grid <= mask_split[1]))
        self.valid_mask2 = ((self.q2_grid >= self.q1_grid - 2 * torch.pi) & 
                            (self.q2_grid <= self.q1_grid - torch.pi))
        self.always_true_mask = torch.ones_like(self.valid_mask, dtype=torch.bool)


        # Flatten for easier computation with vmap
        q_combined = torch.stack((self.q1_grid.flatten(), self.q2_grid.flatten()), dim=-1).to(device)

        # Calculate coordinate change
        Theta1 = mapping_functions[0](q_combined).cpu()
        Theta2 = mapping_functions[1](q_combined).cpu()
        Theta_combined = torch.stack((Theta1,Theta2), dim=-1)

        self.Theta = (Theta1.view(n_lines, n_lines), Theta2.view(n_lines, n_lines))

        
        q1_hat = (mapping_functions[2](Theta_combined.to(device))).cpu()
        q2_hat = (mapping_functions[3](Theta_combined.to(device))).cpu()

        self.q_hat = (q1_hat.view(n_lines, n_lines), q2_hat.view(n_lines, n_lines))

    def make_animation(self, file_name, duration, fps):
        
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

        

        

        # Loop over all frames and morph from original coordinates q to new coordinates theta
        # Append artists of each frame to be animated later
        for frame in range(frames):
            t = torch.tensor(frame / (frames - 1))  # Normalized time from 0 to 1
            plot_func = (torch.sin((2*t-1)*torch.pi/2)+1)/2

            artists = []  # Collect artists for this frame            

            plot_Theta1 = (1 - plot_func) * self.q1_grid + plot_func * self.Theta[0]
            plot_Theta2 = (1 - plot_func) * self.q2_grid + plot_func * self.Theta[1]

                
            for j in range(self.Theta[0].shape[1]):
                valid_indices_hor = self.valid_mask[j, :]
                valid_indices_ver = self.valid_mask[:, j]
                valid_indices_hor2 = self.valid_mask2[j, :]
                valid_indices_ver2 = self.valid_mask2[:, j]
                h_line, = ax.plot(plot_Theta1[j, valid_indices_hor], plot_Theta2[j, valid_indices_hor], color=self.colors[0], lw=0.9)
                v_line, = ax.plot(plot_Theta1[valid_indices_ver, j], plot_Theta2[valid_indices_ver, j], color=self.colors[0], lw=0.9)
                #h_line2, = ax.plot(plot_Theta1[j, valid_indices_hor2], plot_Theta2[j, valid_indices_hor2], color=self.colors[i], lw=0.9)
                #v_line2, = ax.plot(plot_Theta1[valid_indices_ver2, j], plot_Theta2[valid_indices_ver2, j], color=self.colors[i], lw=0.9)
                artists.extend([h_line, v_line])#, h_line2, v_line2])

            frames_artists.insert(frame, artists)


        for frame in range(frames):
            t = torch.tensor(frame / (frames - 1))  # Normalized time from 0 to 1
            plot_func = (torch.sin((2*t-1)*torch.pi/2)+1)/2


            artists = []  # Collect artists for this frame

            plot_q_hat1 = (1 - plot_func) * self.Theta[0] + plot_func * self.q_hat[0]
            plot_q_hat2 = (1 - plot_func) * self.Theta[1] + plot_func * self.q_hat[1]
    

            for j in range(self.Theta[0].shape[1]):
                valid_indices_hor = self.valid_mask[j, :]
                valid_indices_ver = self.valid_mask[:, j]
                valid_indices_hor2 = self.valid_mask2[j, :]
                valid_indices_ver2 = self.valid_mask2[:, j]
                h_line, = ax.plot(plot_q_hat1[j, valid_indices_hor], plot_q_hat2[j, valid_indices_hor], color=self.colors[0], lw=0.9)
                v_line, = ax.plot(plot_q_hat1[valid_indices_ver, j], plot_q_hat2[valid_indices_ver, j], color=self.colors[0], lw=0.9)
                #h_line2, = ax.plot(plot_q_hat1[j, valid_indices_hor2], plot_q_hat2[j, valid_indices_hor2], color=self.colors[i], lw=0.9)
                #v_line2, = ax.plot(plot_q_hat1[valid_indices_ver2, j], plot_q_hat2[valid_indices_ver2, j], color=self.colors[i], lw=0.9)
                artists.extend([h_line, v_line])#, h_line2, v_line2])

            frames_artists.append(artists)
            
        # Duplicate the first and final frame another second to improve viewability
        q_artists = frames_artists[0]
        theta_artists = frames_artists[frames]
        q_hat_artists = frames_artists[-1]
        for frame in range(fps):
            frames_artists.insert(0, q_artists)
            frames_artists.insert(frames+frame, theta_artists)
            frames_artists.append(q_hat_artists)       

        #frames_artists.extend(reversed(frames_artists[:-fps])) 

        # Create the animation
        ani = ArtistAnimation(fig, frames_artists, interval=interval, blit=True)

        # Save the animation as MP4
        print("Creating animation. This may take about a minute")
        ani.save(output_path, writer="ffmpeg", dpi=300, bitrate=-1)
        print("Animation saved to:", output_path)
        plt.close(fig)

    def make_figure(self, file_name):

        # Set the output path
        output_path = "Plotting/Plots/" + file_name

        # Create the figure with corrected settings
        fig, ax = plt.subplots(figsize=(10, 6))


        # Plot the transformed grid lines
        for j in range(self.Theta[0].shape[1]):  # Draw fewer lines for 
            valid_indices_hor = self.valid_mask[j, :]
            valid_indices_ver = self.valid_mask[:, j]
            valid_indices_hor2 = self.valid_mask2[j, :]
            valid_indices_ver2 = self.valid_mask2[:, j]
            ax.plot(self.Theta[0][j, valid_indices_hor], self.Theta[1][j, valid_indices_hor], color=self.colors[0], lw=0.9)
            ax.plot(self.Theta[0][valid_indices_ver, j], self.Theta[1][valid_indices_ver, j], color=self.colors[0], lw=0.9)
            #ax.plot(Theta[0][j, valid_indices_hor2], Theta[1][j, valid_indices_hor2], color=self.colors[i], lw=0.9)
            #ax.plot(Theta[0][valid_indices_ver2, j], Theta[1][valid_indices_ver2, j], color=self.colors[i], lw=0.9)
                

        # Save the figure
        plt.savefig(output_path, dpi=400, bbox_inches='tight', pad_inches=0)
        plt.show(fig)
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