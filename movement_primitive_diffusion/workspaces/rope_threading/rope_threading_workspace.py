from typing import Optional

import numpy as np

from omegaconf import DictConfig

from movement_primitive_diffusion.agents.base_agent import BaseAgent
from movement_primitive_diffusion.workspaces.base_workspace import BaseWorkspace


class RopeThreadingEnvWorkspace(BaseWorkspace):
    def __init__(
        self,
        env_config: DictConfig,
        t_act: int,
        num_upload_successful_videos: int = 5,
        num_upload_failed_videos: int = 5,
        show_images: bool = False,
        seed: Optional[int] = None,
    ):
        self.dt = env_config["control_dt"]

        super().__init__(
            env_config=env_config,
            t_act=t_act,
            num_upload_successful_videos=num_upload_successful_videos,
            num_upload_failed_videos=num_upload_failed_videos,
            show_images=show_images,
        )

        self.seed = seed or np.random.randint(0, 2**32 - 1)
        self.trajectory_seeds: list[np.random.SeedSequence]

        # Discount factor for calculating return values from step rewards
        self.gamma = 1.0

    def render_function(self, caller_locals: dict) -> np.ndarray:
        return self.env._update_rgb_buffer()

    def check_success_hook(self, caller_locals: dict) -> bool:
        return caller_locals["info"]["successful_task"]

    def reset_env(self, caller_locals: dict) -> tuple[np.ndarray, dict]:
        env_index = caller_locals["i"]

        seed = self.trajectory_seeds[env_index]
        obs, info = self.env.reset(seed=seed)

        tip_dist = self.env.get_distance_rope_tip_eyelet()
        self.hooks["min_tip_dist"][env_index] = tip_dist

        self.hooks["tool_pos"][env_index, 0] = info["tool_pose"][:3]
        if self.env.calculate_accelerations:
            self.hooks["rope_accel"][env_index, 0] = info["rope_acceleration"]
            self.hooks["tpsd_accel"][env_index, 0] = info["tool_acceleration"]

        return obs, info

    def post_step_hook(self, caller_locals: dict) -> None:
        env_index = caller_locals["i"]
        info = caller_locals["info"]

        if (tip_dist := self.env.get_distance_rope_tip_eyelet()) < self.hooks["min_tip_dist"][env_index]:
            self.hooks["min_tip_dist"][env_index] = tip_dist

        if (num_pts := self.env.get_num_rope_points_passed_eyelet()) > self.hooks["max_rope_pts_passed"][env_index]:
            self.hooks["max_rope_pts_passed"][env_index] = num_pts

        if (frac_pts := self.env.get_fraction_rope_points_passed_eyelet()) > self.hooks["max_rope_frac_passsed"][env_index]:
            self.hooks["max_rope_frac_passsed"][env_index] = frac_pts

        t = int(self.hooks["episode_lengths"][env_index])
        if not caller_locals["done"]:
            self.hooks["episode_lengths"][env_index] += 1

        self.hooks["reward"][env_index, t+1] = caller_locals["reward"]
        self.hooks["tool_pos"][env_index, t+1] = info["tool_pose"][:3]
        if self.env.calculate_accelerations:
            self.hooks["rope_accel"][env_index, t+1] = info["rope_acceleration"]
            self.hooks["tpsd_accel"][env_index, t+1] =  info["tool_acceleration"]

    def post_episode_hook(self, caller_locals: dict) -> None:
        env_index = caller_locals["i"]

        truncated = caller_locals["truncated"]
        self.hooks["truncation_state"][env_index] = truncated

        tip_dist = self.env.get_distance_rope_tip_eyelet()
        self.hooks["final_tip_dist"][env_index] = tip_dist

        num_pts = self.env.get_num_rope_points_passed_eyelet()
        self.hooks["final_rope_pts_passed"][env_index] = num_pts

        frac_pts = self.env.get_fraction_rope_points_passed_eyelet()
        self.hooks["final_rope_frac_passed"][env_index] = frac_pts

    def test_agent(self, agent: BaseAgent, num_trajectories: int = 10) -> dict:
        # From a fixed start seed, create a seed list of length num_trajectories. These will be used to reset the envs
        seed_sequence = np.random.SeedSequence(self.seed)
        self.trajectory_seeds = seed_sequence.spawn(num_trajectories)

        # Setup numpy arrays that will be updated in the hooks
        self.hooks = {
            "episode_lengths": np.zeros(num_trajectories),
            "truncation_state": np.full(num_trajectories, False),
            "min_tip_dist": np.full(num_trajectories, np.inf),
            "final_tip_dist": np.zeros(num_trajectories),
            "max_rope_pts_passed": np.full(num_trajectories, -np.inf),
            "final_rope_pts_passed": np.zeros(num_trajectories),
            "max_rope_frac_passsed": np.full(num_trajectories, -np.inf),
            "final_rope_frac_passed": np.zeros(num_trajectories),
            "reward": np.full((num_trajectories, self.time_limit + 1), np.nan),
            "tool_pos": np.full((num_trajectories, self.time_limit + 1, 3), np.nan),
        }
        if self.env.calculate_accelerations:
            self.hooks |= {
                "rope_accel": np.full((num_trajectories, self.time_limit + 1), np.nan),
                "tpsd_accel": np.full((num_trajectories, self.time_limit + 1, 4), np.nan),
            }

        # Call the parent's test agent function
        result_dict = super().test_agent(agent, num_trajectories)

        # episode length
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
        final_t = self.hooks["episode_lengths"].astype(int)
        final_rewards = rewards[np.arange(num_trajectories), final_t + 1]
        result_dict["mean_final_reward"] = np.nanmean(final_rewards)

        # tool path length
        cartesian_tool_positions = self.hooks["tool_pos"]
        cartesian_tool_position_deltas = np.diff(cartesian_tool_positions, axis=-2)
        cartesian_tool_path_length = np.nansum(np.linalg.norm(cartesian_tool_position_deltas, axis=-1), axis=-1)
        result_dict["mean_tool_path_length"] = np.mean(cartesian_tool_path_length)
        result_dict["min_tool_path_length"] = np.min(cartesian_tool_path_length)
        result_dict["max_tool_path_length"] = np.max(cartesian_tool_path_length)

        if self.env.calculate_accelerations:
            # rope acceleration
            rope_acceleration = self.hooks["rope_accel"]
            result_dict["mean_rope_acceleration"] = np.nanmean(rope_acceleration)
            result_dict["min_rope_acceleration"] = np.nanmin(rope_acceleration)
            result_dict["max_rope_acceleration"] = np.nanmax(rope_acceleration)

            # rope jerk
            rope_jerk = np.zeros_like(rope_acceleration)
            rope_jerk[:, 1:] = np.abs(np.diff(rope_acceleration, axis=-1)) / self.dt
            result_dict["mean_rope_jerk"] = np.nanmean(rope_jerk)
            result_dict["min_rope_jerk"] = np.nanmin(rope_jerk)
            result_dict["max_rope_jerk"] = np.nanmax(rope_jerk)

            # tool acceleration
            tool_acceleration = np.linalg.norm(self.hooks["tpsd_accel"], axis=-1)
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

        # rope tip distance
        result_dict["mean_min_rope_tip_distance"] = np.mean(self.hooks["min_tip_dist"])
        result_dict["mean_final_rope_tip_distance"] = np.mean(self.hooks["final_tip_dist"])

        # rope points passed
        result_dict["mean_max_rope_points_passed"] = np.mean(self.hooks["max_rope_pts_passed"])
        result_dict["mean_final_rope_points_passed"] = np.mean(self.hooks["final_rope_pts_passed"])

        # rope fraction passed
        result_dict["mean_max_rope_fraction_passed"] = np.mean(self.hooks["max_rope_frac_passsed"])
        result_dict["mean_final_rope_fraction_passed"] = np.mean(self.hooks["final_rope_frac_passed"])

        return result_dict

    def get_result_dict_keys(self) -> list[str]:
        keys = super().get_result_dict_keys()
        keys += [
            "mean_episode_length",
            "truncation_rate",
            "mean_return",
            "min_return",
            "max_return",
            "mean_max_reward",
            "mean_final_reward",
            "mean_tool_path_length",
            "min_tool_path_length",
            "max_tool_path_length",
            "mean_min_rope_tip_distance",
            "mean_final_rope_tip_distance",
            "mean_max_rope_points_passed",
            "mean_final_rope_points_passed",
            "mean_max_rope_fraction_passed",
            "mean_final_rope_fraction_passed",
        ]
        if self.env.calculate_accelerations:
            keys += [
                "mean_rope_acceleration",
                "min_rope_acceleration",
                "max_rope_acceleration",
                "mean_rope_jerk",
                "min_rope_jerk",
                "max_rope_jerk",
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
        return keys
