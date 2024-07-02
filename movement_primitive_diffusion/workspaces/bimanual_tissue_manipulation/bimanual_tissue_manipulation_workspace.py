import numpy as np
import matplotlib.pyplot as plt

from omegaconf import DictConfig
from typing import List, Optional, Dict
import wandb

from movement_primitive_diffusion.agents.base_agent import BaseAgent
from movement_primitive_diffusion.datasets.trajectory_dataset import read_numpy_file
from movement_primitive_diffusion.utils.setup_helper import look_for_trajectory_dir
from movement_primitive_diffusion.workspaces.base_workspace import BaseWorkspace


class BimanualTissueManipulationEnvWorkspace(BaseWorkspace):
    def __init__(
        self,
        env_config: DictConfig,
        t_act: int,
        num_upload_successful_videos: int = 5,
        num_upload_failed_videos: int = 5,
        val_trajectory_dir: Optional[str] = None,
    ):
        super().__init__(
            env_config=env_config,
            t_act=t_act,
            num_upload_successful_videos=num_upload_successful_videos,
            num_upload_failed_videos=num_upload_failed_videos,
        )

        # If we pass a dir that contains a list of trajectories, we will use the target positions from these trajectories to test the agent
        self.val_trajectory_dir = val_trajectory_dir
        if self.val_trajectory_dir is not None:
            self.val_trajectory_dir = look_for_trajectory_dir(self.val_trajectory_dir)
            self.val_trajectories = [traj_dir for traj_dir in self.val_trajectory_dir.iterdir() if traj_dir.is_dir()]
            self.val_trajectories.sort()
            self.target_positions = [read_numpy_file(traj_dir / "target_positions.npz")[0] for traj_dir in self.val_trajectories]

    def reset_env(self, caller_locals: Dict) -> np.ndarray:
        return self.env.reset(options=self.hook_values["options_for_reset"][caller_locals["i"]])

    def render_function(self, caller_locals: Dict) -> np.ndarray:
        return self.env._update_rgb_buffer()

    def post_step_hook(self, caller_locals: Dict) -> None:
        distances = self.env.get_distances_to_targets()["distances"]
        mean_distance = np.mean(distances)
        left_distance = distances[0]
        right_distance = distances[1]

        if mean_distance < self.hook_values["min_distance_per_episode"][caller_locals["i"]]:
            self.hook_values["min_distance_per_episode"][caller_locals["i"]] = mean_distance

        if left_distance < self.hook_values["min_left_distance_per_episode"][caller_locals["i"]]:
            self.hook_values["min_left_distance_per_episode"][caller_locals["i"]] = left_distance

        if right_distance < self.hook_values["min_right_distance_per_episode"][caller_locals["i"]]:
            self.hook_values["min_right_distance_per_episode"][caller_locals["i"]] = right_distance

        self.hook_values["episode_lengths"][caller_locals["i"]] += 1

    def post_episode_hook(self, caller_locals: Dict) -> None:
        distances = self.env.get_distances_to_targets()["distances"]
        mean_distance = np.mean(distances)
        left_distance = distances[0]
        right_distance = distances[1]

        self.hook_values["final_distance_per_episode"][caller_locals["i"]] = mean_distance
        self.hook_values["final_left_distance_per_episode"][caller_locals["i"]] = left_distance
        self.hook_values["final_right_distance_per_episode"][caller_locals["i"]] = right_distance

        self.hook_values["episode_is_successful"][caller_locals["i"]] = caller_locals["successful"]

        caller_locals["pbar"].set_postfix(final_distance=mean_distance)

    def test_agent(self, agent: BaseAgent, num_trajectories: int = 10) -> dict:
        # If num_trajectories is set to -1 and we have a val_trajectory_dir, we will test on all trajectories in this dir
        # Otherwise, we will test on num_trajectories, either from the val_trajectory_dir or randomly generated
        if num_trajectories == -1:
            if self.val_trajectory_dir is not None:
                num_trajectories = len(self.target_positions)
                options_for_reset = [{"target_positions": target_positions} for target_positions in self.target_positions]
            else:
                raise ValueError("If num_trajectories is set to -1, we need to have a val_trajectory_dir.")
        else:
            if self.val_trajectory_dir is not None:
                num_trajectories = min(num_trajectories, len(self.target_positions))
                options_for_reset = [{"target_positions": target_positions} for target_positions in self.target_positions[:num_trajectories]]
            else:
                options_for_reset = [None] * num_trajectories

        # Setup numpy arrays that will be updated in the hooks
        self.hook_values = {
            "options_for_reset": options_for_reset,
            "episode_lengths": np.zeros(num_trajectories),
            "final_distance_per_episode": np.ones(num_trajectories) * np.inf,
            "min_distance_per_episode": np.ones(num_trajectories) * np.inf,
            "final_left_distance_per_episode": np.ones(num_trajectories) * np.inf,
            "min_left_distance_per_episode": np.ones(num_trajectories) * np.inf,
            "final_right_distance_per_episode": np.ones(num_trajectories) * np.inf,
            "min_right_distance_per_episode": np.ones(num_trajectories) * np.inf,
            "episode_is_successful": np.zeros(num_trajectories),
        }

        # Call the parent's test agent function and pass the child's locals() dict
        result_dict = super().test_agent(agent, num_trajectories)

        # Add the additional metrics to the result dict
        result_dict["mean_final_distance"] = np.mean(self.hook_values["final_distance_per_episode"])
        result_dict["mean_min_distance"] = np.mean(self.hook_values["min_distance_per_episode"])
        result_dict["mean_final_left_distance"] = np.mean(self.hook_values["final_left_distance_per_episode"])
        result_dict["mean_min_left_distance"] = np.mean(self.hook_values["min_left_distance_per_episode"])
        result_dict["mean_final_right_distance"] = np.mean(self.hook_values["final_right_distance_per_episode"])
        result_dict["mean_min_right_distance"] = np.mean(self.hook_values["min_right_distance_per_episode"])
        result_dict["mean_episode_length"] = np.mean(self.hook_values["episode_lengths"])

        # Log a bar chart that shows which trajectories were successful and which were not
        fig, ax = plt.subplots()
        bottom = 0
        for i in range(num_trajectories):
            ax.bar("Trajectory", 1, bottom=bottom, color="g" if self.hook_values["episode_is_successful"][i] else "r", edgecolor="black")
            if self.val_trajectory_dir is not None:
                ax.text("Trajectory", bottom + 0.5, self.val_trajectories[i].name, ha="center", va="center", color="white")
            bottom += 1

        # Render plot to numpy array
        fig.canvas.draw()
        image_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_array = image_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        images = wandb.Image(
            image_array,
            caption="Trajectory Success. Green means successful, red means failed.",
        )
        wandb.log({"Trajectory Success": images})

        # Explicitly close the figure to avoid memory leaks
        plt.close(fig)

        return result_dict

    def get_result_dict_keys(self) -> List[str]:
        return super().get_result_dict_keys() + [
            "mean_final_distance",
            "mean_min_distance",
            "mean_final_left_distance",
            "mean_min_left_distance",
            "mean_final_right_distance",
            "mean_min_right_distance",
            "mean_episode_length",
        ]
