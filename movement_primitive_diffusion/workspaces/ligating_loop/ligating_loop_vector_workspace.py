import numpy as np

from omegaconf import DictConfig
from typing import List, Dict

from movement_primitive_diffusion.agents.base_agent import BaseAgent
from movement_primitive_diffusion.workspaces.base_vector_workspace import BaseVectorWorkspace


class LigatingLoopEnvVectorWorkspace(BaseVectorWorkspace):
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
        timeout=5.0,
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
            timeout=timeout,
            annotate_videos=annotate_videos,
        )

        self.dt = self.env_config.time_step * self.env_config.frame_skip
        # WARNING: There is currently a chance, that the env is reset to a state that is invalid (impossible to solve)
        # We thus fix the seed here, and sort out seeds that we manually tested to be invalid.
        self.seed = 42
        seed_sequence = np.random.SeedSequence(self.seed)
        trajectory_seeds = seed_sequence.spawn(110)
        bad_seeds = [100, 83, 82, 56, 53, 27, 21, 8, 6, 3]
        trajectory_seeds = [seed for i, seed in enumerate(trajectory_seeds) if i not in bad_seeds]
        self.trajectory_seeds = trajectory_seeds

        # From a fixed start seed, create a seed list of length num_trajectories. These will be used to reset the envs
        # Discount factor for calculating return values from step rewards
        self.gamma = 1.0

    def check_success_hook(self, caller_locals: Dict) -> bool:
        """Function to modify success check behavior in subclasses.

        For example for checking if the agent reached the goal.

        """
        terminated = caller_locals["env_terminated"][caller_locals["env_index"]]
        truncated = caller_locals["env_truncated"][caller_locals["env_index"]]

        # If the episode is truncated, we set the episode length to the time limit,
        # because the env is either done due to the time limit, or due to truncation in the watchdog
        if truncated:
            env_index_offset = caller_locals["episode_sequence_index"] * self.num_parallel_envs
            trajectory_index = env_index_offset + caller_locals["env_index"]
            if trajectory_index < len(self.trajectory_seeds):
                self.hook_values["episode_lengths"][trajectory_index] = self.time_limit - 1

        return terminated and not truncated

    def reset_env(self, caller_locals: Dict) -> np.ndarray:
        # We initialize this list with None because we might have fewer trajectories left than we have parallel envs
        seeds = [None] * self.num_parallel_envs

        env_index_offset = caller_locals["episode_sequence_index"] * self.num_parallel_envs
        for env_index in range(self.num_parallel_envs):
            trajectory_index = env_index_offset + env_index
            if trajectory_index < len(self.trajectory_seeds):
                seeds[env_index] = self.trajectory_seeds[trajectory_index]
            else:
                break

        return self.vector_env.reset(seed=seeds)

    def render_function(self, caller_locals: Dict) -> np.ndarray:
        return self.vector_env.call("_update_rgb_buffer")

    def post_step_hook(self, caller_locals: Dict) -> None:
        env_index_offset = caller_locals["episode_sequence_index"] * self.num_parallel_envs
        for env_index in range(self.num_parallel_envs):
            trajectory_index = env_index_offset + env_index

            if trajectory_index < caller_locals["num_trajectories"]:
                time_step = int(self.hook_values["episode_lengths"][trajectory_index])
                if not caller_locals["done_buffer"][env_index]:
                    self.hook_values["episode_lengths"][trajectory_index] += 1
                    self.hook_values["tissue_accelerations"][trajectory_index, time_step] = caller_locals["env_info"]["tissue_acceleration"][env_index]
                    self.hook_values["tool_accelerations"][trajectory_index, time_step] = caller_locals["env_info"]["tool_acceleration"][env_index]
                    self.hook_values["loop_accelerations"][trajectory_index, time_step] = caller_locals["env_info"]["loop_acceleration"][env_index]
                    self.hook_values["tool_positions"][trajectory_index, time_step] = caller_locals["env_info"]["tool_positions"][env_index]

                # Extra check to get last reward when env is done
                if time_step <= self.hook_values["episode_lengths"][trajectory_index]:
                    self.hook_values["reward"][trajectory_index, time_step] = caller_locals["env_reward"][env_index]

    def test_agent(self, agent: BaseAgent, num_trajectories: int = 10) -> dict:
        if num_trajectories > len(self.trajectory_seeds):
            raise ValueError(f"Number of trajectories ({num_trajectories}) is larger than the number of seeds ({len(self.trajectory_seeds)})")

        # Setup numpy arrays that will be updated in the hooks
        self.hook_values = {
            "episode_lengths": np.zeros(num_trajectories),
            "tissue_accelerations": np.nan * np.ones((num_trajectories, self.time_limit)),
            "tool_accelerations": np.nan * np.ones((num_trajectories, self.time_limit, 5)),
            "loop_accelerations": np.nan * np.ones((num_trajectories, self.time_limit)),
            "tool_positions": np.nan * np.ones((num_trajectories, self.time_limit, 3)),
            "reward": np.nan * np.ones((num_trajectories, self.time_limit)),
        }

        # Call the parent's test agent function
        result_dict = super().test_agent(agent, num_trajectories)

        # Add the additional metrics to the result dict
        result_dict["mean_episode_length"] = np.mean(self.hook_values["episode_lengths"])
        result_dict["min_episode_length"] = np.min(self.hook_values["episode_lengths"])
        result_dict["max_episode_length"] = np.max(self.hook_values["episode_lengths"])

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

        tool_jerk = np.zeros_like(tool_acceleration)
        tool_jerk[:, 1:] = np.abs(np.diff(tool_acceleration, axis=-1)) / self.dt
        result_dict["mean_tool_jerk"] = np.nanmean(tool_jerk)
        result_dict["min_tool_jerk"] = np.nanmin(tool_jerk)
        result_dict["max_tool_jerk"] = np.nanmax(tool_jerk)

        result_dict["mean_tool_energy"] = np.mean(np.nansum(tool_acceleration, axis=-1))
        result_dict["min_tool_energy"] = np.min(np.nansum(tool_acceleration, axis=-1))
        result_dict["max_tool_energy"] = np.max(np.nansum(tool_acceleration, axis=-1))

        result_dict["mean_loop_acceleration"] = np.nanmean(self.hook_values["loop_accelerations"])
        result_dict["min_loop_acceleration"] = np.nanmin(self.hook_values["loop_accelerations"])
        result_dict["max_loop_acceleration"] = np.nanmax(self.hook_values["loop_accelerations"])

        loop_jerk = np.zeros_like(self.hook_values["loop_accelerations"])
        loop_jerk[:, 1:] = np.abs(np.diff(self.hook_values["loop_accelerations"], axis=-1)) / self.dt
        result_dict["mean_loop_jerk"] = np.nanmean(loop_jerk)
        result_dict["min_loop_jerk"] = np.nanmin(loop_jerk)
        result_dict["max_loop_jerk"] = np.nanmax(loop_jerk)

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

        return result_dict

    def get_result_dict_keys(self) -> List[str]:
        return super().get_result_dict_keys() + [
            "mean_episode_length",
            "min_episode_length",
            "max_episode_length",
            "mean_tissue_acceleration",
            "min_tissue_acceleration",
            "max_tissue_acceleration",
            "mean_tissue_jerk",
            "min_tissue_jerk",
            "max_tissue_jerk",
            "mean_tool_acceleration",
            "min_tool_acceleration",
            "max_tool_acceleration",
            "mean_tool_energy",
            "min_tool_energy",
            "max_tool_energy",
            "mean_tool_jerk",
            "min_tool_jerk",
            "max_tool_jerk",
            "mean_loop_acceleration",
            "min_loop_acceleration",
            "max_loop_acceleration",
            "mean_loop_jerk",
            "min_loop_jerk",
            "max_loop_jerk",
            "mean_tool_path_length",
            "min_tool_path_length",
            "max_tool_path_length",
            "mean_return",
            "min_return",
            "max_return",
        ]
