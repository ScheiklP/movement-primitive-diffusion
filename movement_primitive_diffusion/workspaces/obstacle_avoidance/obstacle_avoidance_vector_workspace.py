from typing import Optional

import numpy as np

from omegaconf import DictConfig

from movement_primitive_diffusion.agents.base_agent import BaseAgent
from movement_primitive_diffusion.workspaces.base_vector_workspace import BaseVectorWorkspace
from movement_primitive_diffusion.workspaces.obstacle_avoidance.obstacle_avoidance_env import (
    Mode,
)
from movement_primitive_diffusion.workspaces.obstacle_avoidance.obstacle_avoidance_utils import (
  plotly_trajectories,
  plotly_trajectory_modes,
)

class ObstacleAvoidanceEnvVectorWorkspace(BaseVectorWorkspace):
    def __init__(
        self,
        env_config: DictConfig,
        t_act: int,
        num_parallel_envs: int,
        shared_memory: bool = False,
        async_vector_env: bool = True,
        num_upload_successful_videos: int = 5,
        num_upload_failed_videos: int = 5,
        video_dt: Optional[float] = None,
        show_images: bool = False,
        annotate_videos: bool = True,
        seed: Optional[int] = None,
    ):
        if video_dt is None:
            video_dt = env_config["control_dt"]

        self.dt = env_config["control_dt"]

        super().__init__(
            env_config=env_config,
            t_act=t_act,
            num_parallel_envs=num_parallel_envs,
            shared_memory=shared_memory,
            async_vector_env=async_vector_env,
            num_upload_successful_videos=num_upload_successful_videos,
            num_upload_failed_videos=num_upload_failed_videos,
            video_dt=video_dt,
            show_images=show_images,
            annotate_videos=annotate_videos,
        )

        self.seed = seed or np.random.randint(0, 2**32 - 1)
        self.trajectory_seeds: list[np.random.SeedSequence]

        # Discount factor for calculating return values from step rewards
        self.gamma = 1.0

    def render_function(self, caller_locals: dict) -> np.ndarray:
        return self.vector_env.call("render")

    def check_success_hook(self, caller_locals: dict) -> bool:
        return caller_locals["env_info"]["success"][caller_locals["env_index"]]

    def reset_env(self, caller_locals: dict) -> tuple[np.ndarray, dict]:
        # We initialize this list with None because we might have fewer trajectories left than we have parallel envs
        seeds = [None for _ in range(self.num_parallel_envs)]

        env_index_offset = caller_locals["episode_sequence_index"] * self.num_parallel_envs
        for env_index in range(self.num_parallel_envs):
            if (traj_idx := env_index_offset + env_index) < len(self.trajectory_seeds):
                seeds[env_index] = self.trajectory_seeds[traj_idx].entropy
            else:
                break

        obs, infos = self.vector_env.reset(seed=seeds)

        for env_index in range(self.num_parallel_envs):
            if (traj_idx := env_index_offset + env_index) < caller_locals["num_trajectories"]:
                self.hooks["eef_pos"][traj_idx, 0] = infos["eef_pos"][env_index][:2]
                self.hooks["eef_vel"][traj_idx, 0] = infos["eef_vel"][env_index][:2]
        return obs, infos


    def post_step_hook(self, caller_locals: dict) -> None:
        rewards = caller_locals["env_reward"]
        infos = caller_locals["env_info"]

        env_index_offset = caller_locals["episode_sequence_index"] * self.num_parallel_envs
        for env_index in range(self.num_parallel_envs):
            if (traj_idx := env_index_offset + env_index) < caller_locals["num_trajectories"]:
                t = int(self.hooks["episode_lengths"][traj_idx])
                if not caller_locals["done_buffer"][env_index]:
                    self.hooks["episode_lengths"][traj_idx] += 1
                # note that we keep overwriting the value at the last timestep when done
                # this works since AsyncVectorEnv does not reset the envs until all are done
                # a finished env just returns the last observation, info, and reward on subsequent steps
                if t <= self.hooks["episode_lengths"][traj_idx]:
                    # t+1 since the first timestep is the initial state, cf. reset
                    self.hooks["reward"][traj_idx, t + 1] = rewards[env_index]
                    self.hooks["eef_pos"][traj_idx, t + 1] = infos["eef_pos"][env_index][:2]
                    self.hooks["eef_vel"][traj_idx, t + 1] = infos["eef_vel"][env_index][:2]

    def post_episode_hook(self, caller_locals: dict) -> None:
        infos = caller_locals["env_info"]
        env_index_offset = caller_locals["episode_sequence_index"] * self.num_parallel_envs
        for env_index in range(self.num_parallel_envs):
            if (traj_idx := env_index_offset + env_index) < caller_locals["num_trajectories"]:
                distance = np.linalg.norm(
                    infos["eef_pos"][env_index][1] - infos["goal_pos"][env_index][1]
                ).mean()
                self.hooks["final_goal_distance"][traj_idx] = distance

                mode = Mode.from_encoding(infos["mode"][env_index])
                self.hooks["final_mode"][traj_idx] = mode

                truncated = caller_locals["env_truncated"][env_index]
                self.hooks["truncation_state"][traj_idx] = truncated

                success = infos["success"][env_index]
                self.hooks["success_state"][traj_idx] = success

    def test_agent(self, agent: BaseAgent, num_trajectories: int = 10) -> dict:
        # From a fixed start seed, create a seed list of length num_trajectories. These will be used to reset the envs
        seed_sequence = np.random.SeedSequence(self.seed)
        self.trajectory_seeds = seed_sequence.spawn(num_trajectories)

        self.hooks = {
            "episode_lengths": np.zeros(num_trajectories),
            "truncation_state": np.full(num_trajectories, False),
            "success_state": np.full(num_trajectories, False),
            "final_goal_distance": np.full(num_trajectories, np.inf),
            "reward": np.full((num_trajectories, self.time_limit + 1), np.nan),
            "eef_pos": np.full((num_trajectories, self.time_limit + 1, 2), np.nan),
            "eef_vel": np.full((num_trajectories, self.time_limit + 1, 2), np.nan),
            "final_mode": [Mode() for _ in range(num_trajectories)],
        }

        # Call the parent's test agent function
        result_dict = super().test_agent(agent, num_trajectories)

        # episode length
        final_t = self.hooks["episode_lengths"].astype(int)
        result_dict["mean_episode_length"] = np.mean(self.hooks["episode_lengths"])

        # truncation rate
        result_dict["truncation_rate"] = np.mean(self.hooks["truncation_state"])

        # return
        rewards = self.hooks["reward"]
        episode_returns = np.array([reward * self.gamma**t for t, reward in enumerate(rewards)])
        result_dict["mean_return"] = np.nanmean(episode_returns)
        result_dict["min_return"] = np.nanmin(episode_returns)
        result_dict["max_return"] = np.nanmax(episode_returns)

        # reward
        result_dict["mean_max_reward"] = np.nanmean(np.nanmax(rewards, axis=-1))
        final_rewards = rewards[np.arange(num_trajectories), final_t + 1]
        result_dict["mean_final_reward"] = np.nanmean(final_rewards)

        # tool path length
        cartesian_tool_positions = self.hooks["eef_pos"]
        cartesian_tool_position_deltas = np.diff(cartesian_tool_positions, axis=-2)
        cartesian_tool_path_length = np.nansum(np.linalg.norm(cartesian_tool_position_deltas, axis=-1), axis=-1)
        result_dict["mean_tool_path_length"] = np.mean(cartesian_tool_path_length)
        result_dict["min_tool_path_length"] = np.min(cartesian_tool_path_length)
        result_dict["max_tool_path_length"] = np.max(cartesian_tool_path_length)

        # tool acceleration
        tool_velocity = self.hooks["eef_vel"]
        tool_acceleration = np.linalg.norm(np.diff(tool_velocity, axis=-2) / self.dt, axis=-1)
        result_dict["mean_tool_acceleration"] = np.nanmean(tool_acceleration)
        result_dict["min_tool_acceleration"] = np.nanmin(tool_acceleration)
        result_dict["max_tool_acceleration"] = np.nanmax(tool_acceleration)

        # tool energy
        tool_energy = np.nansum(tool_acceleration, axis=-1)
        result_dict["mean_tool_energy"] = np.mean(tool_energy)
        result_dict["min_tool_energy"] = np.min(tool_energy)
        result_dict["max_tool_energy"] = np.max(tool_energy)

        # tool jerk
        tool_jerk = np.zeros_like(tool_acceleration)
        tool_jerk[:, 1:] = np.abs(np.diff(tool_acceleration, axis=-1)) / self.dt
        result_dict["mean_tool_jerk"] = np.nanmean(tool_jerk)
        result_dict["min_tool_jerk"] = np.nanmin(tool_jerk)
        result_dict["max_tool_jerk"] = np.nanmax(tool_jerk)

        # goal distance
        result_dict["mean_final_goal_distance"] = np.mean(self.hooks["final_goal_distance"])

        # final modes
        final_modes = self.hooks["final_mode"]
        _, entropy = Mode.compute_distribution(final_modes)
        result_dict["mode_entropy"] = entropy
        result_dict["modes_dec"] = np.array([mode.decode() for mode in final_modes])

        # successful modes
        success_state = self.hooks["success_state"]
        success_modes = [final_modes[i] for i in range(len(final_modes)) if success_state[i]]
        _, success_entropy = Mode.compute_distribution(success_modes)
        result_dict["successful_mode_entropy"] = success_entropy
        result_dict["successful_modes_dec"] = np.array([mode.decode() for mode in success_modes])

        # trajectories
        # strip nan values from trajectories
        failed_trajs = [
            self.hooks["eef_pos"][i, :final_t[i]+1]
            for i in np.arange(num_trajectories)[np.logical_not(success_state)]
        ]
        success_trajs = [
            self.hooks["eef_pos"][i, :final_t[i]+1]
            for i in np.arange(num_trajectories)[success_state]
        ]
        result_dict["failed_trajectories"] = plotly_trajectories(
            trajs=failed_trajs,
            traj_labels=[str(i) for i in np.arange(num_trajectories)[np.logical_not(success_state)]],
            title="Failed trajectories",
        )
        result_dict["successful_trajectories"] = plotly_trajectories(
            trajs=success_trajs,
            traj_labels=[str(i) for i in np.arange(num_trajectories)[success_state]],
            title="Successful trajectories",
        )

        # trajectory modes
        result_dict["trajectory_modes"] = plotly_trajectory_modes(
            traj_modes=final_modes,
            title="Trajectory Modes",
        )

        return result_dict

    def get_result_dict_keys(self) -> list[str]:
        super_keys = super().get_result_dict_keys()
        return super_keys + [
            "mean_episode_length",
            "truncation_rate",
            "mean_return",
            "min_return",
            "max_return",
            "mean_max_reward",
            "mean_final_reward",
            "mean_final_goal_distance",
            "modes_dec",
            "mode_entropy",
            "successful_modes_dec",
            "successful_mode_entropy",
            "failed_trajectories",
            "successful_trajectories",
            "trajectory_modes",
            "mean_tool_path_length",
            "min_tool_path_length",
            "max_tool_path_length",
            "mean_tool_acceleration",
            "min_tool_acceleration",
            "max_tool_acceleration",
            "mean_tool_energy",
            "min_tool_energy",
            "max_tool_energy",
            "mean_tool_jerk",
            "min_tool_jerk",
            "max_tool_jerk",
        ]
