from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
import torch
import numpy as np
import os

class Anim_plotter():

    def __init__(self, rp):
        self.rp = rp
        self.robot_range = 1.2 * (self.rp["l1"] + self.rp["l2"])

        


    def frame_pendulum(self, pos_end, pos_elbow):
        """
        Plot a double pendulum using matplotlib
        Parameters:
        pos_end (torch.Tensor): [x, y] coordinates of the end joint
        pos_elbow (torch.Tensor): [x, y] coordinates of the elbow point
        """
        # Initialize data storage for animation frames
        frames_data = []

        # Shoulder joint is the same for all frames
        shoulder = torch.tensor([0, 0]).numpy()

        for pos_end, pos_elbow in zip(pos_end, pos_elbow):
            # Extract frame data as numpy arrays
            endpoint = pos_end.cpu().detach().numpy()
            elbow = pos_elbow.cpu().detach().numpy()

            # Store data for the current frame
            frames_data.append({
                "links": [
                    {"start": shoulder.tolist(), "end": elbow.tolist()},
                    {"start": elbow.tolist(), "end": endpoint.tolist()}
                ],
                "joints": [
                    {"position": shoulder.tolist()},
                    {"position": elbow.tolist()},
                    {"position": endpoint.tolist()}
                ]
            })

        return frames_data

    def animate_pendulum(self, frames_data_multi, ref_poss=None, plot_actuator = True, file_name="Placeholder.gif", fps=30, dt = 0.01):

        """
        Create and save the double pendulum animation.
        
        Parameters:
            frames_data_multi (list): List of dicts for multiple datasets.
            ref_poss (list): List of dicts of reference positions.
            plot_actuator (bool): Whether to draw the actuator thread.
            file_name (str): Output file name.
            fps (int): Frames per second.
            dt (float): Time step between frames.
        """

        r_outer = self.rp["l1"] + self.rp["l2"]
        if self.rp["l1"] != self.rp["l2"]:
            r_inner = abs(self.rp["l1"] - self.rp["l2"])
        else:
            r_inner = None

        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(script_dir, "..", "Plotting", "Pendulum_plots", file_name)
        print(output_path)

        # Initialize the plot
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(-self.robot_range*1.2, self.robot_range)
        ax.set_ylim(-self.robot_range, self.robot_range*1.2)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Double Pendulum Animation")
        ax.grid(True, linestyle="--", alpha=0.7)

        # Create text element to display the time step
        time_text = ax.text(0.95, 0.95, '', transform=ax.transAxes, ha='right', va='top', fontsize=12, color='black')


        ## Create line and scatter plot elements for the animation
        links_set = []
        joints_set = []
        actuator_set = []
        thread_set = []

        

        for pendulum_data in frames_data_multi:
            ###links_set.append([ax.plot([], [], lw=2, color=frames_data_multi[i][2])[0] for _ in range(2)])
            ###joints_set.append(ax.scatter([], [], s=50, color=frames_data_multi[i][2]))


            # Create two Line2D objects for the two links.
            link_lines = [
                Line2D([], [], lw=2, color=pendulum_data["arm_color"]),
                Line2D([], [], lw=2, color=pendulum_data["arm_color"])
            ]
            for line in link_lines:
                ax.add_line(line)
            links_set.append(link_lines) 

            # Create one Line2D for the joints (using markers only).
            joints_line = Line2D([], [], linestyle='', marker='o',
                                 markersize=8, color=pendulum_data["arm_color"])
            ax.add_line(joints_line)
            joints_set.append(joints_line)            

            if plot_actuator:
                ###x_a, y_a = self.rp["xa"], self.rp["ya"]
                #### Initialise actuator point and thread to end-effector (intensity depending on input?)
                ###actuator_set.append(ax.scatter([], [], s=30, color=frames_data_multi[i][3], alpha=0.8))
                ###thread_set.append(ax.plot([], [], lw=1, color=frames_data_multi[i][3])[0])

                # Actuator marker.
                actuator_line = Line2D([], [], linestyle='', marker='o',
                                       markersize=6, color=pendulum_data["act_color"], alpha=0.8)
                ax.add_line(actuator_line)
                actuator_set.append(actuator_line)
                # Actuator thread from the end effector to a fixed actuator point.
                thread_line = Line2D([], [], lw=1, color=pendulum_data["act_color"])
                ax.add_line(thread_line)
                thread_set.append(thread_line)                

        """
        if ref_pos is not None:
            x_ref, y_ref = ref_pos[0].item(), ref_pos[1].item()
            cross_size = 0.02 * self.robot_range  # Size of cross
            # Reference position cross (two intersecting lines)
            cross = [ax.plot([], [], lw=1, color="black", alpha=0.6)[0] for _ in range(2)]        
        """
        
        ref_crosses = []
        if ref_poss is not None:
            cross_size = 0.025 * self.robot_range  # Determines the size of the cross.
            for ref in ref_poss:
                # Assume that ref["pos"] is a 2-element tensor.
                pos = ref["pos"]
                x_ref, y_ref = pos[0].item(), pos[1].item()
                # Create two lines to form a cross.
                cross1 = Line2D([], [], lw=1.5, color=ref["color"], alpha=0.8)
                cross2 = Line2D([], [], lw=1.5, color=ref["color"], alpha=0.8)
                ax.add_line(cross1)
                ax.add_line(cross2)
                # Set the data so that the lines cross at the target location.
                cross1.set_data([x_ref - cross_size, x_ref + cross_size],
                                [y_ref - cross_size, y_ref + cross_size])
                cross2.set_data([x_ref - cross_size, x_ref + cross_size],
                                [y_ref + cross_size, y_ref - cross_size])
                ref_crosses.append((cross1, cross2))



        # Add circle(s) to the plot to indicate arm radius    
        circle_outer = plt.Circle((0, 0), r_outer, color="gray", fill=False, linestyle="dashed", alpha=0.8)
        ax.add_patch(circle_outer)  

        if r_inner is not None:
            circle_inner = plt.Circle((0, 0), r_inner, color="gray", fill=False, linestyle="dashed", alpha=0.8)
            ax.add_patch(circle_inner)


        # Create legend handles for each dataset
        legend_handles = [Line2D([0], [0], marker="o", color="w", markersize=8, 
                                 markerfacecolor=pend["arm_color"], label=pend["name"])
                        for pend in frames_data_multi]

        if ref_poss is not None:
            ref_handles = [Line2D([0], [0], marker="x", color=ref["color"], markersize=8,
                                linestyle='None', label=ref["name"]) for ref in ref_poss]
            legend_handles.extend(ref_handles)

        # Add legend to the plot
        ax.legend(handles=legend_handles, loc="upper left", title="Datasets")

        def init(): 
            """Initialize animation elements to empty.""" 
                
            for links, joints, in zip(links_set, joints_set):
                for link in links:
                    link.set_data([], [])

                joints.set_data([], [])
            
            for actuator, thread in zip(actuator_set, thread_set):
                if plot_actuator:
                    actuator.set_data([], [])
                    thread.set_data([], [])
                    # Set actuator position
                    ###actuator.set_offsets(np.array([x_a, y_a]))
                    ###thread.set_data([], [])      
            

            time_text.set_text('')

            elements = []
            for link_lines in links_set:
                elements.extend(link_lines)
            elements.extend(joints_set)
            elements.extend(thread_set)
            elements.extend(actuator_set)
            for cross_pair in ref_crosses:
                elements.extend(cross_pair)
            elements.append(time_text)
            
            return elements            


        def update(frame):
            """Update animation elements for a given frame."""

            #links, joints = [], []

            elements = []

            for i, pendulum_data in enumerate(frames_data_multi):
                frames_data = pendulum_data["frames"]
                frame_data = frames_data[frame]
                endpoint = frame_data["links"][1]["end"]

                # Update links
                for j, link_data in enumerate(frame_data["links"]):
                    links_set[i][j].set_data(
                        [link_data["start"][0], link_data["end"][0]],
                        [link_data["start"][1], link_data["end"][1]]
                    )
                # Update joints
                joints = np.array([joint["position"] for joint in frame_data["joints"]]).T
                #joint_positions = np.array([joint["position"] for joint in frame_data["joints"]])
                joints_set[i].set_data(joints[0], joints[1])     
                elements.extend(links_set[i])
                elements.append(joints_set[i]) 

                if plot_actuator:
                    ###thread_set[i].set_data([[endpoint[0], x_a],
                    ###                        [endpoint[1], y_a]])
                    # Update the actuator marker (using fixed actuator point from rp).
                    actuator_set[i].set_data([self.rp["xa"]], [self.rp["ya"]])
                    # Update the thread from the end effector to the actuator point.
                    endpoint = frame_data["links"][1]["end"]
                    thread_set[i].set_data([endpoint[0], self.rp["xa"]],
                                           [endpoint[1], self.rp["ya"]])
                    elements.append(actuator_set[i])
                    elements.append(thread_set[i])    

            for cross_pair in ref_crosses:
                elements.extend(cross_pair)                

            current_time = frame * dt
            time_text.set_text(f"t = {current_time:.2f}s")
            return elements

        # Create the animation
        anim = FuncAnimation(
            fig, update, frames=len(frames_data_multi[0]["frames"]), init_func=init,
            blit=True, interval=1000 // fps
        )

        # Save the animation if a path is provided
        if output_path:
            #anim.save(output_path, fps=fps, extra_args=["-vcodec", "libx264"])
            #anim.save(output_path, writer="pillow", fps=fps)
            anim.save(output_path, writer="ffmpeg", fps=fps, dpi=200)


        plt.show()
