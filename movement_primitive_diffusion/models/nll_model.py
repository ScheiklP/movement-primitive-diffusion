import hydra
import torch

from omegaconf import DictConfig
from typing import Dict, Tuple, Union

from movement_primitive_diffusion.models.base_model import BaseModel
from movement_primitive_diffusion.models.nll_decoder_inner_model import Decoder
from movement_primitive_diffusion.utils.matrix import build_lower_matrix, unsqueeze_and_expand

from torch.distributions import MultivariateNormal

from movement_primitive_diffusion.utils.mp_utils import ProDMPHandler


def nll_loss(true_val: torch.Tensor, pred_mean: torch.Tensor, pred_L: torch.Tensor) -> torch.Tensor:
    """Log likelihood loss

    Args:
        true_val (torch.Tensor): true target values
        pred_mean (torch.Tensor): predicted mean of the Normal distribution
        pred_L (torch.Tensor): alternatively, use predicted Cholesky Decomposition

    Returns:
        log likelihood (torch.Tensor)

    """

    # Construct distribution
    mvn = MultivariateNormal(loc=pred_mean, scale_tril=pred_L, validate_args=False)

    # Compute log likelihood
    ll = mvn.log_prob(true_val).mean()

    # Loss
    ll_loss = -ll
    return ll_loss


class NLLModel(BaseModel):
    def __init__(
        self,
        inner_model_config: DictConfig,
        prodmp_handler_config: DictConfig,
    ):
        super().__init__()

        prodmp_handler_config.num_dof = inner_model_config.action_size
        self.prodmp_handler: ProDMPHandler = hydra.utils.instantiate(prodmp_handler_config)

        inner_model_config.output_size = self.prodmp_handler.encoding_size

        self.decoder: Decoder = hydra.utils.instantiate(inner_model_config)

    def loss(self, state: Tuple[torch.Tensor, torch.Tensor], action: torch.Tensor, extra_inputs: Dict, return_deviations: bool = False) -> torch.Tensor:

        predicted_trajectory, trajectory_uncertainty = self.forward(state, extra_inputs, get_uncertainty=True)

        # change the axis such that value shape and pairs are switched (B, combinations, value shape, 2) -> swith time and value axis
        predicted_trajectory = torch.einsum("...ji->...ij", predicted_trajectory)
        # change the shape from (B, combinations, value shape, 2) to (B, combinations, 2 * value shape)
        # -> (dof1 t1, dof1 t2, dof2 t1, dof2 t2)
        predicted_trajectory = predicted_trajectory.reshape(action.shape)

        # Get Cholesky for covariance
        trajectory_L = torch.linalg.cholesky(trajectory_uncertainty)

        loss = nll_loss(action, predicted_trajectory, trajectory_L)

        if return_deviations:
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
        else:
            return loss

    def forward(self, state: Tuple[torch.Tensor, torch.Tensor], extra_inputs: Dict, get_uncertainty: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        state_mean, state_variance = state

        mean, diagonal, off_diagonal = self.decoder(
            latent_mean=state_mean,
            latent_variance=state_variance,
            additional_inputs=extra_inputs["additional_decoder_inputs"],
        )

        L = build_lower_matrix(diagonal, off_diagonal)

        prediction_times = extra_inputs["prediction_times"]

        if prediction_times.ndim == 3:
            # NOTE: For learning, prediction times has shape (B, combinations, 2),
            # and for prediction in env (B, T)
            combinations = prediction_times.shape[1]

            # Expand dimensions to match all prediction pairs (batch, combinations, ...)
            mean = unsqueeze_and_expand(mean, dim=1, size=combinations)
            L = unsqueeze_and_expand(L, dim=1, size=combinations)

        trajectory, trajectory_positions_L = self.prodmp_handler.decode(
            params=mean,
            params_L=L,
            initial_time=extra_inputs["initial_time"],
            initial_position=extra_inputs["initial_position"],
            initial_velocity=extra_inputs["initial_velocity"],
            times=prediction_times,
        )

        if get_uncertainty:
            return trajectory, trajectory_positions_L
        else:
            return trajectory
