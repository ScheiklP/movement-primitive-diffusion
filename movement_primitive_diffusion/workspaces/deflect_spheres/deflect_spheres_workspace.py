import numpy as np
from typing import List, Dict
from movement_primitive_diffusion.agents.base_agent import BaseAgent
from movement_primitive_diffusion.workspaces.base_workspace import BaseWorkspace


class DeflectSpheresEnvWorkspace(BaseWorkspace):
    def reset_env(self, caller_locals: Dict) -> np.ndarray:
        reset_result = self.env.reset()
        self.hook_values["min_distance_per_episode"][caller_locals["i"]] = self.env.get_current_distance_tool_active_sphere()
        return reset_result

    def post_step_hook(self, caller_locals: Dict) -> None:
        current_distance = self.env.get_current_distance_tool_active_sphere()
        previous_min_distance = self.hook_values["min_distance_per_episode"][caller_locals["i"]]

        if current_distance < previous_min_distance:
            self.hook_values["min_distance_per_episode"][caller_locals["i"]] = current_distance

        self.hook_values["episode_lengths"][caller_locals["i"]] += 1

    def post_episode_hook(self, caller_locals: Dict) -> None:
        caller_locals["pbar"].set_postfix(min_distance=self.hook_values["min_distance_per_episode"][caller_locals["i"]])

    def render_function(self, caller_locals: Dict) -> np.ndarray:
        return self.env._update_rgb_buffer()

    def test_agent(self, agent: BaseAgent, num_trajectories: int = 10) -> dict:
        self.hook_values = {
            "min_distance_per_episode": np.ones(num_trajectories) * np.inf,
            "episode_lengths": np.zeros(num_trajectories),
        }

        # Call the parent's test agent function
        result_dict = super().test_agent(agent, num_trajectories)

        # Add the min distance and episode length to the result dict
        result_dict["mean_min_distance"] = np.mean(self.hook_values["min_distance_per_episode"])
        result_dict["mean_episode_length"] = np.mean(self.hook_values["episode_lengths"])

        return result_dict

    def get_result_dict_keys(self) -> List[str]:
        super_keys = super().get_result_dict_keys()
        return super_keys + ["mean_min_distance", "mean_episode_length"]
