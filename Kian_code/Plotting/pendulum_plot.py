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


    def frame_pendulum(self, pos_end, pos_elbow, colors = None, x_des = None, height = 6, width = 6):
        """
        Plot a double pendulum using matplotlib
        
        Parameters:
        x_elbow (torch.Tensor): [x, y] coordinates of the elbow joint
        x (torch.Tensor): [x, y] coordinates of the end point
        """
        
        if colors is None:
            colors = ["orange" for i in range(pos_end.size(0))]

        # Initialize data storage for animation frames
        frames_data = []

        # Shoulder joint is the same for all frames
        shoulder = torch.tensor([0, 0]).numpy()

        for pos_end, pos_elbow, color in zip(pos_end, pos_elbow, colors):

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
                ],
                "color": color
            })

        return frames_data

    def animate_pendulum(self, frames_data_multi, ref_pos=None, actuator = True, file_name="Placeholder.gif", fps=30):



        r_outer = self.rp["l1"] + self.rp["l2"]
        if self.rp["l1"] != self.rp["l2"]:
            r_inner = abs(self.rp["l1"] - self.rp["l2"])
        else:
            r_inner = None

        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(script_dir, "..", "Plotting", "Pendulum_plots", file_name)
        print(output_path)
        """
        Create and optionally save the double pendulum animation.

        Parameters:
        file_name (str, optional): File path to save the animation (e.g., "animation.mp4").
        fps (int, optional): Frames per second for the animation. Default is 30.
        """
        # Initialize the plot
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(-self.robot_range*1.2, self.robot_range)
        ax.set_ylim(-self.robot_range, self.robot_range*1.2)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Double Pendulum Animation")
        ax.grid(True, linestyle="--", alpha=0.7)

        ## Create line and scatter plot elements for the animation
        #links = [ax.plot([], [], lw=2, color="orange")[0] for _ in range(2)]
        #joints = ax.scatter([], [], s=50, color="orange")

        links_set = []
        joints_set = []
        actuator_set = []
        thread_set = []

        colors_actuator = ["tab:red", "tab:cyan"]

        for i in range(len(frames_data_multi)):
            links_set.append([ax.plot([], [], lw=2, color=frames_data_multi[i][2])[0] for _ in range(2)])
            joints_set.append(ax.scatter([], [], s=50, color=frames_data_multi[i][2]))

            if actuator:
                x_a, y_a = self.rp["xa"], self.rp["ya"]
                # Initialise actuator point and thread to end-effector (intensity depending on input?)
                actuator_set.append(ax.scatter([], [], s=30, color=frames_data_multi[i][3], alpha=0.8))
                thread_set.append(ax.plot([], [], lw=1, color=frames_data_multi[i][3])[0])

        if ref_pos is not None:
            x_ref, y_ref = ref_pos[0].item(), ref_pos[1].item()
            cross_size = 0.02 * self.robot_range  # Size of cross
            # Reference position cross (two intersecting lines)
            cross = [ax.plot([], [], lw=1, color="black", alpha=0.6)[0] for _ in range(2)]        




        # Add circle(s) to the plot to indicate arm radius    
        circle_outer = plt.Circle((0, 0), r_outer, color="gray", fill=False, linestyle="dashed", alpha=0.8)
        ax.add_patch(circle_outer)  

        if r_inner is not None:
            circle_inner = plt.Circle((0, 0), r_inner, color="gray", fill=False, linestyle="dashed", alpha=0.8)
            ax.add_patch(circle_inner)


        # Create legend handles for each dataset
        legend_handles = [Line2D([0], [0], marker="o", color="w", markersize=8, markerfacecolor=color, label=name)
                        for i, (_, name, color, _) in enumerate(frames_data_multi)]

        # Add legend to the plot
        ax.legend(handles=legend_handles, loc="upper left", title="Datasets")

        def init(): 
            """Initialize animation elements to empty."""          
            for links, joints, actuator, thread in zip(links_set, joints_set, actuator_set, thread_set):
                for link in links:
                    link.set_data([], [])

                joints.set_offsets(np.empty((0, 2))) 

                if actuator:
                    # Set actuator position
                    actuator.set_offsets(np.array([x_a, y_a]))
                    thread.set_data([], [])      

            if ref_pos is not None:
                # Set reference position as cross
                cross[0].set_data([x_ref - cross_size, x_ref + cross_size], [y_ref - cross_size, y_ref + cross_size])
                cross[1].set_data([x_ref - cross_size, x_ref + cross_size], [y_ref + cross_size, y_ref - cross_size])

            return links + [joints] + thread_set + (cross if ref_pos is not None else [])#, thread]


        def update(frame):
            """Update animation elements for a given frame."""

            links, joints = [], []

            for i, frames_tuple in enumerate(frames_data_multi):
                frames_data = frames_tuple[0]
                frame_data = frames_data[frame]
                endpoint = frame_data["links"][1]["end"]
                # Update links
                for j, link_data in enumerate(frame_data["links"]):
                    links_set[i][j].set_data(
                        [link_data["start"][0], link_data["end"][0]],
                        [link_data["start"][1], link_data["end"][1]]
                    )
                # Update joints
                joint_positions = np.array([joint["position"] for joint in frame_data["joints"]])
                joints_set[i].set_offsets(joint_positions)      

                links.append(links_set[i][j]) 
                joints.append(joints_set[i])     

                if actuator:
                    thread_set[i].set_data([[endpoint[0], x_a],
                                            [endpoint[1], y_a]])

            return links + joints + (thread_set if actuator else [])

        # Create the animation
        anim = FuncAnimation(
            fig, update, frames=len(frames_data_multi[0][0]), init_func=init,
            blit=True, interval=1000 // fps
        )

        # Save the animation if a path is provided
        if output_path:
            #anim.save(output_path, fps=fps, extra_args=["-vcodec", "libx264"])
            #anim.save(output_path, writer="pillow", fps=fps)
            anim.save(output_path, writer="ffmpeg", fps=fps, dpi=300)


        plt.show()
