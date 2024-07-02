import numpy as np

from typing import List, Dict

from movement_primitive_diffusion.agents.base_agent import BaseAgent
from movement_primitive_diffusion.workspaces.base_vector_workspace import BaseVectorWorkspace


class DeflectSpheresEnvVectorWorkspace(BaseVectorWorkspace):
    def reset_env(self, caller_locals: Dict) -> np.ndarray:
        reset_result = self.vector_env.reset()
        env_index_offset = caller_locals["episode_sequence_index"] * self.num_parallel_envs
        current_distances = self.vector_env.call("get_current_distance_tool_active_sphere")
        for env_index in range(self.num_parallel_envs):
            trajectory_index = env_index_offset + env_index
            if trajectory_index < caller_locals["num_trajectories"]:
                self.hook_values["min_distance_per_episode"][trajectory_index] = current_distances[env_index]
            else:
                break
        return reset_result

    def render_function(self, caller_locals: Dict) -> np.ndarray:
        return self.vector_env.call("_update_rgb_buffer")

    def post_step_hook(self, caller_locals: Dict) -> None:
        current_distances = self.vector_env.call("get_current_distance_tool_active_sphere")

        env_index_offset = caller_locals["episode_sequence_index"] * self.num_parallel_envs
        for env_index in range(self.num_parallel_envs):
            trajectory_index = env_index_offset + env_index

            if trajectory_index < caller_locals["num_trajectories"]:
                previous_min_distance = self.hook_values["min_distance_per_episode"][trajectory_index]
                if current_distances[env_index] < previous_min_distance:
                    self.hook_values["min_distance_per_episode"][trajectory_index] = current_distances[env_index]

                if not caller_locals["done_buffer"][env_index]:
                    self.hook_values["episode_lengths"][trajectory_index] += 1

    def test_agent(self, agent: BaseAgent, num_trajectories: int = 10) -> dict:
        # Setup numpy arrays that will be updated in the hooks
        self.hook_values = {
            "min_distance_per_episode": np.ones(num_trajectories) * np.inf,
            "episode_lengths": np.zeros(num_trajectories),
        }

        # Call the parent's test agent function
        result_dict = super().test_agent(agent, num_trajectories)

        # Add the additional metrics to the result dict
        result_dict["mean_min_distance"] = np.mean(self.hook_values["min_distance_per_episode"])
        result_dict["mean_episode_length"] = np.mean(self.hook_values["episode_lengths"])

        return result_dict

    def get_result_dict_keys(self) -> List[str]:
        super_keys = super().get_result_dict_keys()
        return super_keys + ["mean_min_distance", "mean_episode_length"]
