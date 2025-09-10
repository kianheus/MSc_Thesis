import torch
import os
import json
import Double_Pendulum.transforms as transforms
import Plotting.pendulum_plot as pendulum_plot


def create_video(rp, dt, t_end, plotter, stride, q_multiseries, xy_des, save_dir, video_caption):

	"Obtain REAL trajectories for 3 kinds of simulation"
	q_nn_series, q_ana_series, q_naive_series = q_multiseries	

	frames_data = []

	if q_naive_series is not None:
		pos_end_q_naive, pos_elbow_q_naive = torch.vmap(transforms.forward_kinematics, in_dims=(None, 0))(rp, q_naive_series[::stride])
		frames_q_naive = plotter.frame_pendulum(pos_end_q_naive, pos_elbow_q_naive)
		frames_data.append({
			"frames": frames_q_naive,
			"times": dt,
			"name": "naive",
			"arm_color": "tab:green", 
			"act_color": "tab:olive"
		})

	if q_nn_series is not None:
		pos_end_q_nn, pos_elbow_q_nn = torch.vmap(transforms.forward_kinematics, in_dims=(None, 0))(rp, q_nn_series[::stride])
		frames_q_nn = plotter.frame_pendulum(pos_end_q_nn, pos_elbow_q_nn)

		frames_data.append({
			"frames": frames_q_nn,
			"times": dt,
			"name": "collocated", 
			"arm_color": "tab:orange", #blue
			"act_color": "tab:red" #cyan
		})

	if q_ana_series is not None:
		pos_end_q_ana, pos_elbow_q_ana = torch.vmap(transforms.forward_kinematics, in_dims=(None, 0))(rp, q_ana_series[::stride])
		frames_q_ana = plotter.frame_pendulum(pos_end_q_ana, pos_elbow_q_ana)
		frames_data.append({
			"frames": frames_q_ana,
			"times": dt,
			"name": "q_est",
			"arm_color": "tab:blue", 
			"act_color": "tab:cyan"
		})



	ref_pos_real = {
		"pos": xy_des,
		"name": "reference",
		"color": "tab:blue"
	}

	ref_poss = [ref_pos_real]#, ref_pos_est]#, ref_pos_naive]


	name_rp = "RP:(" + str(rp["xa"]) + "," + str(rp["ya"]) + ")_"
	name_ref = "ref:(" + str(xy_des[0].item()) + "," + str(xy_des[1].item()) + ")_"

	file_name = "t_end:[" + str(t_end) + "]_dt:[" + str(dt) + "]_stride:[" + str(stride) + "]_0.mp4"
	file_counter = 0

	output_path = os.path.join(save_dir, file_name)

	while os.path.isfile(output_path):
		print("file name already exists")
		file_counter += 1
		file_name = file_name[:-6] + "_" + str(file_counter) + ".mp4"
		output_path = os.path.join(save_dir, file_name)

	plotter.animate_pendulum(frames_data, video_caption, ref_poss=ref_poss, plot_actuator=True, save_path=output_path, fps = 1/(dt*stride), dt = dt*stride)
	#plotter.animate_pendulum(frames_data, ref_pos=None, plot_actuator=False, file_name=file_name, fps = 1/(dt*stride), dt = dt*stride)

def save_metadata(rp, dt, t_end, timestamp, xy_des, xy_start, Kp, Kd, sim_type, use_neural_net, model_cw, model_location, save_dir):
	
	os.makedirs(save_dir, exist_ok=True)
	metadata = {
		"timestamp": timestamp,
		"Learned transform": use_neural_net,
		"Model location": model_location,
		"Model clockwise": model_cw, 
		"PD gains": {
			"Kp": [[Kp[0,0].item(), Kp[0,1].item()], 
				[Kp[1,0].item(), Kp[1,1].item()]],  
			"Kd": [[Kd[0,0].item(), Kd[0,1].item()], 
				[Kd[1,0].item(), Kd[1,1].item()]]   
		},
		"Actuator location": {"x_a": rp["xa"], "y_a": rp["ya"]},
		"xy start": {"x": xy_start[0].item(), "y": xy_start[1].item()},
		"xy des": {"x": xy_des[0].item(), "y": xy_des[1].item()},
		"Time step": dt,
		"Sim time": t_end,
	}

	metadata["PD gains"]["Kp"] = str(metadata["PD gains"]["Kp"])
	metadata["PD gains"]["Kd"] = str(metadata["PD gains"]["Kd"])

	metadata_path = os.path.join(save_dir, "metadata.json")
	with open(metadata_path, "w") as f:
		json.dump(metadata, f, indent=4)
		  
def save_trajectory_plots(rp, dt, t_end, xy_des, q_des, th_des, save_dir, xy_multiseries, q_multiseries, th_multiseries):
	
	xy_nn_series, xy_ana_series, xy_naive_series = xy_multiseries
	q_nn_series, q_ana_series, q_naive_series = q_multiseries
	th_ana_nn_series, th_ana_ana_series, th_ana_naive_series = th_multiseries

	datasets_q = []
	if q_nn_series is not None:
		datasets_q.append({
				"name": "collocated",
				"values": q_nn_series.cpu().detach().numpy(),
				"color": "tab:orange",
				"style": "solid"
			})
	if q_ana_series is not None:
		datasets_q.append({
				"name": "Analytic",
				"values": q_ana_series.cpu().detach().numpy(),
				"color": "tab:blue",
				"style": "dotted"
			})
	if q_naive_series is not None:
		datasets_q.append({
			    "name": "naive",
		    	"values": q_naive_series.cpu().detach().numpy(),
		    	"color": "tab:green",
				"style": "dashed"
			})
		
	datasets_th = []
	if th_ana_nn_series is not None:
		datasets_th.append({
				"name": "collocated",
				"values": th_ana_nn_series.cpu().detach().numpy(),
				"color": "tab:orange",
				"style": "solid"
		})
	if th_ana_ana_series is not None:
		datasets_th.append({
				"name": "Analytic",
				"values": th_ana_ana_series.cpu().detach().numpy(),
				"color": "tab:blue",
				"style": "dotted"
		})
	if th_ana_naive_series is not None:
		datasets_th.append({
				"name": "naive",
				"values": th_ana_naive_series.cpu().detach().numpy(),
				"color": "tab:green",
				"style": "dashed"
		})

	datasets_xy = []
	if xy_nn_series is not None:
		datasets_xy.append({
				"name": "Learned",
				"values": xy_nn_series.cpu().detach().numpy(),
				"color": "tab:orange",
				"style": "solid"
		})
	if xy_ana_series is not None:
		datasets_xy.append({
				"name": "Analytic",
				"values": xy_ana_series.cpu().detach().numpy(),
				"color": "tab:blue",
				"style": "dotted"
		})
	if xy_naive_series is not None:
		datasets_xy.append({
				"name": "Naive",
				"values": xy_naive_series.cpu().detach().numpy(),
				"color": "tab:green",
				"style": "dashdot"
		})



	# Common labels for the plots.
	name_q = r"$q$" + "-space"
	name_th = r"$\theta$" + "-space"
	name_xy = r"$xy$" + "-space trajectory"
	t_series = torch.arange(0, t_end, dt)

	# Create an instance of ErrorPlotter.
	ep = pendulum_plot.Error_plotter_alt(rp)

	# Prepare plot datasets for each column.
	# Each call groups a set of datasets to be drawn in one subplot column.
	column1 = ep.create_plot_dataset(t=t_series, datasets=datasets_q, reference=q_des, name=name_q)
	column2 = ep.create_plot_dataset(t=t_series, datasets=datasets_th, reference=th_des, name=name_th)
	column3 = ep.create_plot_dataset(t=t_series, datasets=datasets_xy, reference=xy_des.unsqueeze(0), name=name_xy)
	plot_datasets = [column1, column2]#, column3]

	file_name = "Error plot.png"
	file_counter = 0

	legend_locations = [["lower right", "upper right"], ["lower right", "upper right"]]


	output_path = os.path.join(save_dir, file_name)

	# Pass the list of columns (plot_dataset objects) to plot_multi.
	axes_names = [[r"$q_0$" + " " + r"$(rad)$", r"$q_1$" + " " + r"$(rad)$"], 
			      [r"$\theta_a$" + " " + r"$(m)$", r"$\theta_u$" + " " + r"$(rad)$"], 
				  [r"$x$" + " " + r"$(m)$", r"$y$" + " " + r"$(m)$"]]
	ep.plot_multi(plot_datasets=plot_datasets, save_path=output_path, axes_names = axes_names, empty = True, legend_locations = legend_locations)
	ep.plot_multi(plot_datasets=plot_datasets, save_path=output_path, axes_names = axes_names, empty = False, legend_locations = legend_locations)