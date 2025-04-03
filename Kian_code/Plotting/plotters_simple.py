import torch
from torch import Tensor
from matplotlib import pyplot as plt
import Learning.training_data as training_data
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os

"""
def plot_data(device, points_tensor):
    q1_plot = torch.linspace(training_data.q1_low, training_data.q1_high, 50)
    q2_plot = torch.linspace(training_data.q2_low, training_data.q2_high, 50)

    q1_plot = points_tensor[:, 0].cpu()
    q2_plot = points_tensor[:, 1].cpu()
    #q1_grid, q2_grid = torch.meshgrid(q1_plot, q2_plot, indexing='ij')
    plot_points = torch.stack([q1_plot, q2_plot], dim=-1).to(device)  # Shape: (N, 2)
    #print(plot_points.size())
    #return plot_points, q1_grid, q2_grid
    return plot_points
"""


def plot_3d_double(points_tensor, zs, plot_title, sub_titles, xlabel, ylabel, zlabel, folder_path):
     
    x = points_tensor[:,0].cpu()
    y = points_tensor[:,1].cpu()

    file_name = plot_title.replace(" ", "_")
    file_path = os.path.join(folder_path, file_name)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), subplot_kw={'projection': '3d'})

    for i in range(2):
        z = zs[i].cpu()
        sc = axes[i].scatter(x, y, z, c=z, cmap="viridis", edgecolor='none')
        axes[i].set_xlabel(xlabel)
        axes[i].set_ylabel(ylabel)
        axes[i].set_zlabel(zlabel)
        axes[i].set_title(sub_titles[i])
        divider = make_axes_locatable(axes[i])
        cbar_ax = fig.add_axes([0.42+0.47*i, 0.19, 0.01, 0.64])
        cbar = fig.colorbar(sc, cax=cbar_ax, orientation='vertical')
    fig.suptitle(plot_title, fontsize=16, y=0.95)  # General title
    fig.subplots_adjust(left=0.0, right=0.9, wspace=0.1)  
    #plt.tight_layout

    plt.savefig(file_path)
    plt.show()   

def plot_3d_quad(points_tensor, zs, plot_title, sub_titles, xlabel, ylabel, zlabel, folder_path):

    x = points_tensor[:, 0].cpu()
    y = points_tensor[:, 1].cpu()

    file_name = plot_title.replace(" ", "_")
    file_path = os.path.join(folder_path, file_name)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8.5), subplot_kw={'projection': '3d'})
    for i in range(2):
        for j in range(2):

            z = zs[2*i + j].cpu()
            sc = axes[i, j].scatter(x, y, z, c=z, cmap="viridis", edgecolor="none")
            axes[i, j].set_xlabel(xlabel)
            axes[i, j].set_ylabel(ylabel)
            axes[i, j].set_zlabel(zlabel)
            axes[i, j].set_title(sub_titles[2*i+j])
            cbar_ax = fig.add_axes([0.42+0.47*j, 0.57 - 0.42*i, 0.01, 0.3])
            fig.colorbar(sc, cax=cbar_ax, orientation='vertical')
    
    fig.suptitle(plot_title, fontsize=16, y=0.95)  
    fig.subplots_adjust(left=0.0, right=0.9, wspace=0.1)  
    plt.tight_layout
    plt.savefig(file_path)
    plt.show()
