import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

import Plotting.plotters_simple as plotters_simple

import training_data

from matplotlib.patches import Circle
import warnings
warnings.filterwarnings("ignore")

def training_points_plotter(points, extend = None, save = False, save_folder = None):

	""" 
	Simple plotter function which visualizes the points used for training of the Autoencoder. 
	"""
	 
	if save_folder is not None:
		save_path = save_folder + "/training_points.eps"
	else:
		save_path = None

	plt.figure(figsize=(6, 6))
	plt.scatter(points[:, 0].cpu().numpy(), points[:, 1].cpu().numpy(), alpha=0.6, edgecolors='k', s=20)
	plt.title('Scatter Plot of q0 vs q1')
	plt.xlabel('q0')
	plt.ylabel('q1')
	plt.xlim(-2*torch.pi, 2*torch.pi)
	plt.ylim(-2*torch.pi, 2*torch.pi)
	plt.grid(True)
	
	if save:
		plt.savefig(save_path, format="eps")
	plt.show()


def plot_loss(train_loss, val_loss, file_counter, log = False, save_folder = None):

	if save_folder is not None:
		save_path = save_folder + "/loss_vs_epoch_" + str(file_counter) + ".pdf"
	else:
		save_path = None

	"""
	Plots training and validation loss. 
	ylim" and "yscale" should be enabled depending on the loss function.
	"""

	plt.figure(figsize=(10, 6))
	plt.plot(train_loss, label="Training Loss")
	plt.plot(val_loss, label="Validation Loss")
	plt.xlabel("Epoch")
	plt.ylabel("Loss")
	#plt.ylim((-1, 40))
	plt.legend()
	plt.title("Training and Validation Loss over Epochs")
	plt.grid(True)
	if log:
		plt.yscale("log")
	if save_path is not None:
		plt.savefig(save_path, format="pdf")
	plt.show()
	



def plot_losses_vs_epoch(train_losses, val_losses, save_folder = None):
	"""
	Plot training & validation losses over epochs for multiple runs.
	
	- Semi-transparent thin lines for each individual run
	- Bold line = mean across runs
	- Shaded band = ±1 standard deviation
	"""

	if save_folder is not None:
		save_path = save_folder + "/losses_vs_epoch.pdf"
	else:
		save_path = None
	
	
	# Assume every run has the same number of epochs
	epochs = np.arange(0, len(train_losses[0]))
	train_losses = np.vstack(train_losses)  # shape = (n_runs, n_epochs)
	val_losses   = np.vstack(val_losses)
	
	# Compute mean & std
	mean_train = train_losses.mean(axis=0)
	std_train  = train_losses.std(axis=0)
	mean_val   = val_losses.mean(axis=0)
	std_val    = val_losses.std(axis=0)
	
	plt.figure(figsize=(8, 6))
	
	# Plot each individual run (train & val) with low alpha
	for i in range(train_losses.shape[0]):
		plt.plot(epochs, train_losses[i], color='C0', alpha=0.3, linewidth=1)
		plt.plot(epochs, val_losses[i],   color='C1', alpha=0.3, linewidth=1, linestyle='--')
	
	# Plot mean ± std shading
	plt.plot(epochs, mean_train, color='C0', label='Train mean', linewidth=2)
	plt.fill_between(epochs,
					 mean_train - std_train,
					 mean_train + std_train,
					 color='C0', alpha=0.2,
					 label='Train ±1σ')

	plt.plot(epochs, mean_val, color='C1', label='Val mean', linewidth=2, linestyle='--')
	plt.fill_between(epochs,
					 mean_val - std_val,
					 mean_val + std_val,
					 color='C1', alpha=0.2,
					 label='Val ±1σ')
	
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.title('Training & Validation Loss Across Runs')
	plt.grid(True)
	plt.legend()
	plt.yscale("log")
	plt.tight_layout()
	if save_path is not None:
		plt.savefig(save_path, format="pdf")
	plt.show()


def check_clockwise_vectorized(q):
		"""
		Expects q to be a tensor of shape (N,2) where each row is [q0, q1].
		Returns two boolean masks: (cw_mask, ccw_mask), where:
			- cw_mask[i] is True if the i-th configuration is elbow clockwise.
			- ccw_mask[i] is True if the i-th configuration is elbow counterclockwise.
		
		The logic is as follows (from your original function):
			If q1 lies between q0 and q0+π, or between q0-2π and q0-π, then the configuration
			is considered counterclockwise. Otherwise it is clockwise.
		"""
		q0 = q[:, 0]
		q1 = q[:, 1]
		cond_ccw = ((q1 >= q0) & (q1 <= q0 + torch.pi))
		cw_mask = ~cond_ccw
		ccw_mask = cond_ccw
		return cw_mask, ccw_mask


def plot_yinyang(n_points, save_folder, file_counter, train_clockwise, models, model_names, rp, device):
	for i in range(2):

		if i == 0:
			# Create 1D tensors for q0 and q1 in the range [-pi, pi]
			q0_vals = torch.linspace(-np.pi, np.pi, n_points)
			q1_vals = torch.linspace(-np.pi, 2*np.pi, n_points)

			theta_xy_fig_path = os.path.join(save_folder, "theta_vs_q_full" + str(file_counter) + ".eps")
		if i == 1:
			q0_vals = torch.linspace(training_data.q0_low, training_data.q0_high, n_points)
			q1_vals = torch.linspace(training_data.q1_low, training_data.q1_high, n_points)		
			theta_xy_fig_path = os.path.join(save_folder, "theta_vs_q_partial" + str(file_counter) + ".eps")

		# Create a 2D grid (meshgrid) of q values.
		# (Note: using indexing='ij' so that the first axis corresponds to q0 and the second to q1)
		q0_grid, q1_grid = torch.meshgrid(q0_vals, q1_vals, indexing='ij')

		# Stack the grid to get a tensor of shape (n_points*n_points, 2)
		q_grid = torch.stack([q0_grid.flatten(), q1_grid.flatten()], dim=1).to(device)

		# === Compute theta0 and theta1 using the analytic encoder functions ===
		# We use torch.vmap to evaluate the functions over the batch of q values.
		# Note: encoder_theta_0_ana and encoder_theta_1_ana each return a tuple (theta, theta).

		if train_clockwise:
				raise ValueError("This plotter currently only supports plotting counterclockwise performance. Should be an easy fix")


		fig, axes = plt.subplots(2, 2, figsize=(10, 8.5))
		for i, (model, model_name) in enumerate(zip(models, model_names)):

				theta_out = model.encoder_vmap(q_grid)
				#theta_out = torch.vmap(model.encoder)(q_grid)

				theta0 = theta_out[:, 0]
				theta1 = theta_out[:, 1]

				# Since q1_grid and q2_grid are already on a mesh, we can compute x_end and y_end elementwise.
				x_end = rp["l0"] * torch.cos(q_grid[:, 0]) + rp["l1"] * torch.cos(q_grid[:, 1])
				y_end = rp["l0"] * torch.sin(q_grid[:, 0]) + rp["l1"] * torch.sin(q_grid[:, 1])

				# --- Determine configuration (clockwise vs. counterclockwise) for each q ---
				cw_mask, ccw_mask = check_clockwise_vectorized(q_grid)
				# Counterclockwise points
				x_end_ccw   = x_end[ccw_mask].detach().cpu().numpy()
				y_end_ccw   = y_end[ccw_mask].detach().cpu().numpy()
				theta0_ccw  = theta0[ccw_mask].detach().cpu().numpy()
				theta1_ccw  = theta1[ccw_mask].detach().cpu().numpy()
				thetas = [theta0_ccw, theta1_ccw]

				l_total = rp["l0"] + rp["l1"]

				for j in range(2):
						sc = axes[i, j].scatter(x_end_ccw, y_end_ccw, c=thetas[j], cmap='viridis', s=5)

						circle = Circle((0.0, 0.0),
										l_total*1.01,
										fill=False,
										linestyle='--',
										edgecolor='k',
										linewidth=1.0)
						axes[i, j].add_patch(circle)					
						axes[i, j].set_title(r"$\theta_{" + str(j) + "}$" + " - " + model_names[i])
						axes[i, j].set_xlabel("x")
						axes[i, j].set_ylabel("y")
						cbar = plt.colorbar(sc, ax=axes[i, j])
						cbar.set_label(r'$\theta$', fontsize=14)
						axes[i, j].set_xlim((-l_total*1.02, l_total*1.02))
						axes[i, j].set_ylim((-l_total*1.02, l_total*1.02))

		plt.tight_layout()
		plt.savefig(theta_xy_fig_path, format="eps")
		plt.show()


def plot_model_performance(model, model_ana, plot_dataloader, save_folder, device):

	os.makedirs(save_folder, exist_ok=True)

	model.eval()
	with torch.no_grad():
		for (q, M_q, A_q) in plot_dataloader:
			q = q.to(device)
			M_q = M_q.to(device)
			A_q = A_q.to(device)

			theta, J_h, q_hat, J_h_dec, J_h_ana = model(q)
			theta_ana = model.theta_ana(q)
			J_h_trans = torch.transpose(J_h, 1, 2)
			J_h_inv = J_h_dec
			J_h_inv_trans = torch.transpose(J_h_inv, 1, 2)

			J_h_inv_ana = torch.linalg.inv(J_h_ana)
			J_h_inv_trans_ana = torch.transpose(J_h_inv_ana, 1, 2)

			M_th = J_h_inv_trans @ M_q @ J_h_inv
			A_th = (J_h_inv_trans @ A_q).squeeze(-1)

			M_th_ana = J_h_inv_trans_ana @ M_q @ J_h_inv_ana
			A_th_ana = (J_h_inv_trans_ana @ A_q).squeeze(-1)


			A_th_cpu = A_th.cpu().detach().numpy()
			print("Percentage of abs(A_0) > 0.6:", 100 * np.sum(np.abs(A_th_cpu[:, 0]) > 0.6)/A_th_cpu[:, 0].size, "%")
			print("Percentage of abs(A_1) < 0.3:", 100 * np.sum(np.abs(A_th_cpu[:, 1]) < 0.3)/A_th_cpu[:, 1].size, "%")
			
			
			M_th_cpu = M_th.cpu().detach().numpy()
			print("Percentage of abs(M_00) > 1.0:", 100 * np.sum(np.abs(M_th_cpu[:, 0, 0]) > 1.0)/M_th_cpu[:, 0, 0].size, "%")
			print("Percentage of abs(M_01) < 0.2:", 100 * np.sum(np.abs(M_th_cpu[:, 0, 1]) < 0.2)/M_th_cpu[:, 0, 1].size, "%")
			print("Percentage of abs(M_11) > 1.0:", 100 * np.sum(np.abs(M_th_cpu[:, 1, 1]) > 1.0)/M_th_cpu[:, 1, 1].size, "%")

			plotting_3d = False
			plotting_2d = True
			if plotting_3d:
				plotters_simple.plot_3d_quad(q, [theta_ana[:, 0], theta[:, 0], theta_ana[:, 1], theta[:, 1]], "Analytic vs learned theta", 
											["th0_ana", "th0_learned", "th_1_ana", "th_1_learned"], "q_0", "q_1", "th", save_folder)
				plotters_simple.plot_3d_double(q, [A_th[:, 0], A_th[:, 1]], "Input decoupling", ["A0", "A1"], "q_0", "q_1", "A", save_folder)
				plotters_simple.plot_3d_quad(q, [M_th[:, 0, 0], M_th[:, 0, 1], M_th[:, 1, 0], M_th[:, 1, 1]], "M_th vs q", 
										 ["M_th[0,0]", "M_th[0,1]", "M_th[1,0]", "M_th[1,1]"], "q_0", "q_1", "M_th", save_folder)
				#plotters_simple.plot_3d_double(q, J_h_ana[:, 0, 0], J_h[:, 0, 1], "J_h", "00", "01", "q_0", "q_1", "J_h", device)
				#plotters_simple.plot_3d_double(q, J_h_ana[:, 1, 0], J_h[:, 1, 1], "J_h", "10", "11", "q_0", "q_1", "J_h", device)			
			
			if plotting_2d:
				plotters_simple.plot_2d_double(q, [A_th[:, 0], A_th[:, 1]], "Input matrix terms", [r"$A_{0}$", r"$A_{1}$"], r"$q_0$", r"$q_1$", r"$A$", save_folder)
				plotters_simple.plot_2d_quad(q, [M_th[:, 0, 0], M_th[:, 0, 1], M_th[:, 1, 0], M_th[:, 1, 1]], r"$M_{\theta}$" + " vs " + r"$q$", 
										 [r"$M_{\theta_{0,0}}$", r"$M_{\theta_{0,1}}$", r"$M_{\theta_{1,0}}$", r"$M_{\theta_{1,1}}$"], r"$q_{0}$", r"$q_{1}$", "M_th", save_folder)
				
			


		