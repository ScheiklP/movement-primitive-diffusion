import numpy as np
import wandb
import math
import matplotlib.pyplot as plt
import hydra

from omegaconf import DictConfig
from typing import List, Optional, Dict
from pathlib import Path

from movement_primitive_diffusion.agents.base_agent import BaseAgent
from movement_primitive_diffusion.workspaces.base_vector_workspace import BaseVectorWorkspace
from movement_primitive_diffusion.datasets.trajectory_dataset import read_numpy_file
from movement_primitive_diffusion.utils.setup_helper import look_for_trajectory_dir


class BimanualTissueManipulationEnvVectorWorkspace(BaseVectorWorkspace):
    def __init__(
        self,
        env_config: DictConfig,
        t_act: int,
        num_parallel_envs: int,
        shared_memory: bool = False,
        async_vector_env: bool = True,
        num_upload_successful_videos: int = 5,
        num_upload_failed_videos: int = 5,
        show_images: bool = False,
        val_trajectory_dir: Optional[str] = None,
        annotate_videos: bool = True,
    ):
        super().__init__(
            env_config=env_config,
            t_act=t_act,
            num_parallel_envs=num_parallel_envs,
            shared_memory=shared_memory,
            async_vector_env=async_vector_env,
            num_upload_successful_videos=num_upload_successful_videos,
            num_upload_failed_videos=num_upload_failed_videos,
            video_dt=env_config.time_step * env_config.frame_skip,
            show_images=show_images,
            annotate_videos=annotate_videos,
        )

        self.dt = self.env_config.time_step * self.env_config.frame_skip

        # Discount factor for calculating return values from step rewards
        self.gamma = 1.0

        # If we pass a dir that contains a list of trajectories, we will use the target positions from these trajectories to test the agent
        self.val_trajectory_dir = val_trajectory_dir
        if self.val_trajectory_dir is not None:
            self.val_trajectory_dir = look_for_trajectory_dir(self.val_trajectory_dir)
            self.val_trajectories = [traj_dir for traj_dir in self.val_trajectory_dir.iterdir() if traj_dir.is_dir()]
            self.val_trajectories.sort()
            self.target_positions = [read_numpy_file(traj_dir / "target_positions.npz")[0] for traj_dir in self.val_trajectories]

    def reset_env(self, caller_locals: Dict) -> np.ndarray:
        # We initialize this list with None because we might have fewer trajectories left than we have parallel envs
        per_env_options = [None] * self.num_parallel_envs

        env_index_offset = caller_locals["episode_sequence_index"] * self.num_parallel_envs
        for env_index in range(self.num_parallel_envs):
            trajectory_index = env_index_offset + env_index
            if trajectory_index < len(self.hook_values["options_for_reset"]):
                per_env_options[env_index] = self.hook_values["options_for_reset"][trajectory_index]
            else:
                break

        return self.vector_env.reset(per_env_options=per_env_options)

    def render_function(self, caller_locals: Dict) -> np.ndarray:
        return self.vector_env.call("_update_rgb_buffer")

    def post_step_hook(self, caller_locals: Dict) -> None:
        per_env_distances = self.vector_env.call("get_distances_to_targets")

        env_index_offset = caller_locals["episode_sequence_index"] * self.num_parallel_envs
        for env_index in range(self.num_parallel_envs):
            trajectory_index = env_index_offset + env_index

            if trajectory_index < caller_locals["num_trajectories"]:
                distances = per_env_distances[env_index]["distances"]
                mean_distance = np.mean(distances)
                left_distance = distances[0]
                right_distance = distances[1]

                if mean_distance < self.hook_values["min_distance_per_episode"][trajectory_index]:
                    self.hook_values["min_distance_per_episode"][trajectory_index] = mean_distance

                if left_distance < self.hook_values["min_left_distance_per_episode"][trajectory_index]:
                    self.hook_values["min_left_distance_per_episode"][trajectory_index] = left_distance

                if right_distance < self.hook_values["min_right_distance_per_episode"][trajectory_index]:
                    self.hook_values["min_right_distance_per_episode"][trajectory_index] = right_distance

                time_step = int(self.hook_values["episode_lengths"][trajectory_index])
                if not caller_locals["done_buffer"][env_index]:
                    self.hook_values["episode_lengths"][trajectory_index] += 1
                    self.hook_values["tissue_accelerations"][trajectory_index, time_step] = caller_locals["env_info"]["tissue_acceleration"][env_index]
                    self.hook_values["tool_accelerations"][trajectory_index, time_step] = caller_locals["env_info"]["tool_acceleration"][env_index]

                    self.hook_values["tool_positions"][trajectory_index, time_step] = caller_locals["env_info"]["tool_positions"][env_index]

                # Extra check to get last reward when env is done
                if time_step <= self.hook_values["episode_lengths"][trajectory_index]:
                    self.hook_values["reward"][trajectory_index, time_step] = caller_locals["env_reward"][env_index]

    def post_episode_hook(self, caller_locals: Dict) -> None:
        per_env_distances = self.vector_env.call("get_distances_to_targets")

        env_index_offset = caller_locals["episode_sequence_index"] * self.num_parallel_envs
        for env_index in range(self.num_parallel_envs):
            trajectory_index = env_index_offset + env_index
            if trajectory_index < caller_locals["num_trajectories"]:
                distances = per_env_distances[env_index]["distances"]
                mean_distance = np.mean(distances)
                left_distance = distances[0]
                right_distance = distances[1]

                self.hook_values["final_distance_per_episode"][trajectory_index] = mean_distance
                self.hook_values["final_left_distance_per_episode"][trajectory_index] = left_distance
                self.hook_values["final_right_distance_per_episode"][trajectory_index] = right_distance
                self.hook_values["episode_is_successful"][trajectory_index] = caller_locals["successful_buffer"][env_index]

                self.progress_bar.set_postfix(final_distance=mean_distance)

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
                assert num_trajectories is not None, f"If we have a val_trajectory_dir, we need to set num_trajectories to -1 or a positive integer. Received {num_trajectories}"
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
            "tissue_accelerations": np.nan * np.ones((num_trajectories, self.time_limit)),
            "tool_accelerations": np.nan * np.ones((num_trajectories, self.time_limit, 4)),
            "tool_positions": np.nan * np.ones((num_trajectories, self.time_limit, 6)),
            "reward": np.nan * np.ones((num_trajectories, self.time_limit)),
        }

        # Call the parent's test agent function
        result_dict = super().test_agent(agent, num_trajectories)

        # Add the additional metrics to the result dict
        result_dict["mean_final_distance"] = np.mean(self.hook_values["final_distance_per_episode"])
        result_dict["mean_min_distance"] = np.mean(self.hook_values["min_distance_per_episode"])

        result_dict["mean_final_left_distance"] = np.mean(self.hook_values["final_left_distance_per_episode"])
        result_dict["mean_min_left_distance"] = np.mean(self.hook_values["min_left_distance_per_episode"])

        result_dict["mean_final_right_distance"] = np.mean(self.hook_values["final_right_distance_per_episode"])
        result_dict["mean_min_right_distance"] = np.mean(self.hook_values["min_right_distance_per_episode"])

        result_dict["mean_episode_length"] = np.mean(self.hook_values["episode_lengths"])

        result_dict["mean_tissue_acceleration"] = np.nanmean(self.hook_values["tissue_accelerations"])
        result_dict["min_tissue_acceleration"] = np.nanmin(self.hook_values["tissue_accelerations"])
        result_dict["max_tissue_acceleration"] = np.nanmax(self.hook_values["tissue_accelerations"])
        tissue_jerk = np.zeros_like(self.hook_values["tissue_accelerations"])
        tissue_jerk[:, 1:] = np.abs(np.diff(self.hook_values["tissue_accelerations"], axis=-1)) / self.dt
        result_dict["mean_tissue_jerk"] = np.nanmean(tissue_jerk)
        result_dict["min_tissue_jerk"] = np.nanmin(tissue_jerk)
        result_dict["max_tissue_jerk"] = np.nanmax(tissue_jerk)

        tool_acceleration = np.linalg.norm(self.hook_values["tool_accelerations"], axis=-1)
        result_dict["mean_tool_acceleration"] = np.nanmean(tool_acceleration)
        result_dict["min_tool_acceleration"] = np.nanmin(tool_acceleration)
        result_dict["max_tool_acceleration"] = np.nanmax(tool_acceleration)

        tool_energy = np.nansum(tool_acceleration, axis=-1)
        result_dict["mean_tool_energy"] = np.mean(tool_energy)
        result_dict["min_tool_energy"] = np.min(tool_energy)
        result_dict["max_tool_energy"] = np.max(tool_energy)

        tool_jerk = np.zeros_like(tool_acceleration)
        tool_jerk[:, 1:] = np.abs(np.diff(tool_acceleration, axis=-1)) / self.dt
        result_dict["mean_tool_jerk"] = np.nanmean(tool_jerk)
        result_dict["min_tool_jerk"] = np.nanmin(tool_jerk)
        result_dict["max_tool_jerk"] = np.nanmax(tool_jerk)

        cartesian_tool_positions = self.hook_values["tool_positions"]
        cartesian_tool_position_deltas = np.diff(cartesian_tool_positions, axis=-2)
        cartesian_tool_path_length = np.nansum(np.linalg.norm(cartesian_tool_position_deltas, axis=-1), axis=-1)
        result_dict["mean_tool_path_length"] = np.mean(cartesian_tool_path_length)
        result_dict["min_tool_path_length"] = np.min(cartesian_tool_path_length)
        result_dict["max_tool_path_length"] = np.max(cartesian_tool_path_length)

        rewards = self.hook_values["reward"]
        return_values = np.array([reward * self.gamma**t for t, reward in enumerate(rewards)])
        result_dict["mean_return"] = np.nanmean(return_values)
        result_dict["min_return"] = np.nanmin(return_values)
        result_dict["max_return"] = np.nanmax(return_values)

        # Log a bar chart that shows which trajectories were successful and which were not
        fig, ax = plt.subplots()

        num_sequential_episodes = math.ceil(num_trajectories / self.num_parallel_envs)
        start_indices = [self.num_parallel_envs * offset for offset in range(0, num_sequential_episodes)]
        end_indices = [min(start_index + self.num_parallel_envs, num_trajectories) for start_index in start_indices]
        for start, end in zip(start_indices, end_indices):
            x_location = f"Trajectories {start} to {end}"
            bottom = 0
            for trajectory_index in range(start, end):
                ax.bar(x_location, 1, bottom=bottom, color="g" if self.hook_values["episode_is_successful"][trajectory_index] else "r", edgecolor="black")
                if self.val_trajectory_dir is not None:
                    ax.text(x_location, bottom + 0.5, self.val_trajectories[trajectory_index].name, ha="center", va="center", color="white")
                bottom += 1

        # Render plot to numpy array
        fig.canvas.draw()
        image_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_array = image_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        images = wandb.Image(
            image_array,
            caption="Trajectory Success. Green means successful, red means failed.",
        )

        if wandb.run is not None:
            wandb.log({"Trajectory Success": images})
        else:
            base_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir) / "plots"
            base_dir.mkdir(parents=True, exist_ok=True)
            existing_plots = len(list(base_dir.glob("*.png")))
            plt.savefig(base_dir / f"trajectory_success_{existing_plots}.png")

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
            "mean_tissue_acceleration",
            "min_tissue_acceleration",
            "max_tissue_acceleration",
            "mean_tool_acceleration",
            "min_tool_acceleration",
            "max_tool_acceleration",
            "mean_tool_energy",
            "min_tool_energy",
            "max_tool_energy",
            "mean_tool_jerk",
            "min_tool_jerk",
            "max_tool_jerk",
            "mean_tissue_jerk",
            "min_tissue_jerk",
            "max_tissue_jerk",
            "mean_tool_path_length",
            "min_tool_path_length",
            "max_tool_path_length",
            "mean_return",
            "min_return",
            "max_return",
        ]
