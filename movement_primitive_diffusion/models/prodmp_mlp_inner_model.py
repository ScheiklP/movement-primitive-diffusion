import hydra
import torch

from omegaconf import DictConfig
from typing import Dict, Union, Optional

from movement_primitive_diffusion.models.base_inner_model import BaseInnerModel
from movement_primitive_diffusion.networks.mlp import SimpleMLP
from movement_primitive_diffusion.utils.mp_utils import ProDMPHandler


class ProDMPMLPInnerModel(BaseInnerModel):
    def __init__(
        self,
        state_size: int,
        action_size: int,
        t_pred: int,
        num_layers: int,
        num_neurons: int,
        prodmp_handler_config: DictConfig,
        sigma_embedding_config: DictConfig,
        hidden_nonlinearity: Union[str, torch.nn.Module] = "LeakyReLU",  # Module, not Functional.
        min_tau: Optional[float] = None,
        max_tau: Optional[float] = None,
    ):
        super().__init__()

        # TODO should we move this into the ProDMPHandler?
        # Adapt tau_factor, if tau is learned
        if prodmp_handler_config.learn_tau:
            if min_tau is None or max_tau is None:
                raise ValueError("If learn_tau is True, min_tau and max_tau must be provided.")
            if min_tau >= max_tau:
                raise ValueError("min_tau must be smaller than max_tau.")

            # Set tau_factor such that tau during ProDMP initializtation represents tau_max.
            tau_factor = max_tau / (prodmp_handler_config.dt * prodmp_handler_config.traj_steps)
            prodmp_handler_config.tau_factor = tau_factor

        # Initialize ProDMPHandler
        prodmp_handler_config.num_dof = action_size
        self.prodmp_handler: ProDMPHandler = hydra.utils.instantiate(prodmp_handler_config)

        # Add a variable to hold the latest predicted ProDMP parameters
        self.latest_prodmp_parameters: Union[None, torch.Tensor] = None

        # Sigma embedding
        self.sigma_embedding: torch.nn.Module = hydra.utils.instantiate(sigma_embedding_config)
        dummy_sigma = torch.zeros((1, 1))
        embedded_sigma = self.sigma_embedding(dummy_sigma)

        # get nonlinearity
        if isinstance(hidden_nonlinearity, str):
            hidden_nonlinearity: torch.nn.Module = getattr(torch.nn, hidden_nonlinearity)

        # size of encoded observation + size of one action times the number of time steps to predict + size of sigma embedding
        input_size = state_size + action_size * t_pred + embedded_sigma.shape[-1]

        self.mlp_model: SimpleMLP = SimpleMLP(
            input_size=input_size,
            num_layers=num_layers,
            num_neurons=num_neurons,
            hidden_nonlinearity=hidden_nonlinearity,
            output_size=self.prodmp_handler.encoding_size,
        )
        # With learn_tau, learn_delay, learn_alpha_phase: params = [tau, delay, alpha_phase, dof weights, goal]
        # TODO should we move this into the ProDMPHandler?
        self.relu = torch.nn.ReLU()
        self.min_tau = min_tau
        self.max_tau = max_tau

        self.input_size = input_size
        self.output_size = action_size * t_pred

    def forward(self, state: torch.Tensor, noised_action: torch.Tensor, sigma: torch.Tensor, extra_inputs: Dict) -> torch.Tensor:
        minibatch_size = state.shape[0]
        sigma = sigma.view(minibatch_size, -1)
        embedded_sigma = self.sigma_embedding(sigma)
        flat_noised_action = noised_action.view(minibatch_size, -1)
        prodmp_params = self.mlp_model(torch.cat([state, flat_noised_action, embedded_sigma], dim=-1))

        # TODO should we move this into the ProDMPHandler?
        iterator = 0
        if self.prodmp_handler.learn_tau:
            assert self.min_tau is not None and self.max_tau is not None, f"If learn_tau is True, min_tau and max_tau must be provided. Got {self.min_tau=} and {self.max_tau=}."
            sigmoid_result = torch.sigmoid(prodmp_params[:, iterator]) * (self.max_tau - self.min_tau) + self.min_tau
            prodmp_params = torch.cat((sigmoid_result.unsqueeze(-1), prodmp_params[:, iterator + 1 :]), dim=-1)
            iterator += 1
        if self.prodmp_handler.learn_delay:
            relu_result = self.relu(prodmp_params[:, iterator])
            prodmp_params = torch.cat((relu_result.unsqueeze(-1), prodmp_params[:, iterator + 1 :]), dim=-1)
            iterator += 1
        if self.prodmp_handler.learn_alpha_phase:
            relu_result = self.relu(prodmp_params[:, iterator])
            prodmp_params = torch.cat((relu_result.unsqueeze(-1), prodmp_params[:, iterator + 1 :]), dim=-1)
            iterator += 1

        if iterator > 0:
            # If any of the hyperparameters are learnable, reset the MP
            self.prodmp_handler.mp.reset()

        # Set the latest prodmp parameters so that they can be used in predict_upsampled
        self.latest_prodmp_parameters = prodmp_params

        trajectory = self.prodmp_handler.decode(prodmp_params, **extra_inputs)

        return trajectory
