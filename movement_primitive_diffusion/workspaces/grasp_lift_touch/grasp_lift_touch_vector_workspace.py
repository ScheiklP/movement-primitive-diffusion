import numpy as np

from omegaconf import DictConfig
from typing import List, Optional, Dict

from movement_primitive_diffusion.agents.base_agent import BaseAgent
from movement_primitive_diffusion.workspaces.base_vector_workspace import BaseVectorWorkspace


class GraspLiftTouchEnvVectorWorkspace(BaseVectorWorkspace):
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
        seed: Optional[int] = None,
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
        self.seed = seed or np.random.randint(0, 2**32 - 1)
        self.trajectory_seeds: List[np.random.SeedSequence]
        # Discount factor for calculating return values from step rewards
        self.gamma = 1.0

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
                    info = caller_locals["env_info"]
                    self.hook_values["episode_lengths"][trajectory_index] += 1
                    self.hook_values["tissue_accelerations"][trajectory_index, time_step] = info["tissue_acceleration"][env_index]
                    self.hook_values["tool_accelerations"][trajectory_index, time_step] = info["tool_acceleration"][env_index]
                    self.hook_values["collisions"][trajectory_index, time_step] = info["collision_cauter_gripper"][env_index] + info["collision_cauter_gallbladder"][env_index] + info["collision_cauter_liver"][env_index] + info["collision_gripper_liver"][env_index]
                    self.hook_values["tool_positions"][trajectory_index, time_step] = caller_locals["env_info"]["tool_positions"][env_index]

                # Extra check to get last reward when env is done
                if time_step <= self.hook_values["episode_lengths"][trajectory_index]:
                    self.hook_values["reward"][trajectory_index, time_step] = caller_locals["env_reward"][env_index]

    def post_episode_hook(self, caller_locals: Dict) -> None:
        env_index_offset = caller_locals["episode_sequence_index"] * self.num_parallel_envs
        for env_index in range(self.num_parallel_envs):
            trajectory_index = env_index_offset + env_index
            if trajectory_index < caller_locals["num_trajectories"]:
                self.hook_values["final_phase"][trajectory_index] = caller_locals["env_info"]["final_phase"][env_index]

                # self.progress_bar.set_postfix(final_distance=mean_distance)

    def test_agent(self, agent: BaseAgent, num_trajectories: int = 10) -> dict:
        # From a fixed start seed, create a seed list of length num_trajectories. These will be used to reset the envs
        seed_sequence = np.random.SeedSequence(self.seed)
        self.trajectory_seeds = seed_sequence.spawn(num_trajectories)

        # Setup numpy arrays that will be updated in the hooks
        self.hook_values = {
            "episode_lengths": np.zeros(num_trajectories),
            "final_phase": np.zeros(num_trajectories),
            "tissue_accelerations": np.nan * np.ones((num_trajectories, self.time_limit)),
            "tool_accelerations": np.nan * np.ones((num_trajectories, self.time_limit, 9)),
            "collisions": np.nan * np.ones((num_trajectories, self.time_limit)),
            "tool_positions": np.nan * np.ones((num_trajectories, self.time_limit, 6)),
            "reward": np.nan * np.ones((num_trajectories, self.time_limit)),
        }

        # Call the parent's test agent function
        result_dict = super().test_agent(agent, num_trajectories)

        # Add the additional metrics to the result dict
        result_dict["mean_episode_length"] = np.mean(self.hook_values["episode_lengths"])
        result_dict["min_episode_length"] = np.min(self.hook_values["episode_lengths"])
        result_dict["max_episode_length"] = np.max(self.hook_values["episode_lengths"])

        result_dict["mean_final_phase"] = np.mean(self.hook_values["final_phase"])
        result_dict["min_final_phase"] = np.min(self.hook_values["final_phase"])
        result_dict["max_final_phase"] = np.max(self.hook_values["final_phase"])

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

        result_dict["mean_collisions"] = np.nanmean(self.hook_values["collisions"])
        result_dict["min_collisions"] = np.nanmin(self.hook_values["collisions"])
        result_dict["max_collisions"] = np.nanmax(self.hook_values["collisions"])

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
            "mean_final_phase",
            "min_final_phase",
            "max_final_phase",
            "mean_tissue_acceleration",
            "min_tissue_acceleration",
            "max_tissue_acceleration",
            "mean_tool_acceleration",
            "min_tool_acceleration",
            "max_tool_acceleration",
            "mean_tool_energy",
            "min_tool_energy",
            "max_tool_energy",
            "mean_collisions",
            "min_collisions",
            "max_collisions",
            "mean_tissue_jerk",
            "min_tissue_jerk",
            "max_tissue_jerk",
            "mean_tool_jerk",
            "min_tool_jerk",
            "max_tool_jerk",
            "mean_tool_path_length",
            "min_tool_path_length",
            "max_tool_path_length",
            "mean_return",
            "min_return",
            "max_return",
        ]
