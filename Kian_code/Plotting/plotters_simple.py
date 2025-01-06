import torch
from torch import Tensor
from matplotlib import pyplot as plt
import Learning.training_data as training_data

def plot_data(device):
    q1_plot = torch.linspace(training_data.q1_low, training_data.q1_high, 50)
    q2_plot = torch.linspace(training_data.q2_low, training_data.q2_high, 50)
    q1_grid, q2_grid = torch.meshgrid(q1_plot, q2_plot, indexing='ij')
    plot_points = torch.stack([q1_grid.flatten(), q2_grid.flatten()], dim=-1).to(device)  # Shape: (N, 2)
    return plot_points, q1_grid, q2_grid

def plot_3d_double(z1, z2, plot_title, title_1, title_2, zlabel, device, z_limits = None):
     # Create side-by-side plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), subplot_kw={'projection': '3d'})
    plt.subplots_adjust(top=0.85)
    fig.suptitle(plot_title, fontsize=16, y=0.85)  # General title

    plot_points, x, y = plot_data(device)
    z1 = z1.view((50, 50)).cpu().numpy()
    z2 = z2.view((50, 50)).cpu().numpy()

    xlabel = "$q_1$ (rad)"
    ylabel = "$q_2$ (rad)"

    # Left plot: Analytic function
    axes[0].plot_surface(x, y, z1, cmap='viridis', edgecolor='none')
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel(ylabel)
    axes[0].set_zlabel(zlabel)
    axes[0].set_title(title_1)

    # Apply z-limits if provided
    if z_limits is not None:
        axes[0].set_zlim(z_limits)

    # Right plot: Learned function
    surf = axes[1].plot_surface(x, y, z2, cmap='plasma', edgecolor='none')
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel(ylabel)
    axes[1].set_zlabel(zlabel)
    axes[1].set_title(title_2)

    # Add colorbar for both plots
    fig.colorbar(surf, ax=axes, shrink=0.5, aspect=15, orientation='vertical', label=zlabel)
    plt.show()   