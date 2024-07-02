import hydra
import torch

from typing import Dict, Tuple, Union

from omegaconf import DictConfig


class GMMModel(torch.nn.Module):
    def __init__(
        self,
        inner_model_config: DictConfig,
    ):
        super().__init__()
        self.inner_model = hydra.utils.instantiate(inner_model_config)

    def loss(self, state: torch.Tensor, action: torch.Tensor, extra_inputs: Dict = None, return_deviations: bool = False) -> torch.Tensor:
        predicted_action_dists:torch.distributions.MixtureSameFamily = self.inner_model(state)
        log_probs = predicted_action_dists.log_prob(action)

        loss = -torch.mean(log_probs)

        if return_deviations:
            predicted_trajectory = predicted_action_dists.sample()
            # Calculate the L2 error of the first and last action of the sequence
            # The first combination of times holds t1 and t2 -> even indices of that are dof1 t1, dof2 t1, ... -> start of the trajectory
            start = action[:, 0, 0::2]
            predicted_start = predicted_trajectory[:, 0, 0::2]

            # The last combination of times holds tN-1 and tN -> odd indices of that are dof1 tN, dof2 tN, ... -> end of the trajectory
            end = action[:, -1, 1::2]
            predicted_end = predicted_trajectory[:, -1, 1::2]

            start_point_deviation = torch.linalg.norm(start - predicted_start, dim=-1).mean()
            end_point_deviation = torch.linalg.norm(end - predicted_end, dim=-1).mean()

            return loss, start_point_deviation, end_point_deviation


        return loss


    def forward(self, state: torch.Tensor, extra_inputs: Dict = None) -> torch.Tensor:
        predicted_action_dists = self.inner_model(state)
        predicted_action = predicted_action_dists.sample()

        return predicted_action


