import torch
import hydra

from omegaconf import DictConfig
from typing import Dict

from movement_primitive_diffusion.networks.mlp import SimpleMLP
from movement_primitive_diffusion.models.base_inner_model import BaseInnerModel


class MLPInnerModel(BaseInnerModel):
    def __init__(
        self,
        state_size: int,
        action_size: int,
        t_pred: int,
        num_layers: int,
        num_neurons: int,
        sigma_embedding_config: DictConfig,
        activation_function: str = "ReLU",
    ) -> None:
        super().__init__()

        self.sigma_embedding: torch.nn.Module = hydra.utils.instantiate(sigma_embedding_config)
        dummy_sigma = torch.zeros((1, 1))
        embedded_sigma = self.sigma_embedding(dummy_sigma)

        # size of encoded observation + size of one action times the number of time steps to predict + size of sigma embedding
        input_size = state_size + action_size * t_pred + embedded_sigma.shape[-1]

        self.model = SimpleMLP(
            input_size=input_size,
            num_layers=num_layers,
            num_neurons=num_neurons,
            output_size=action_size * t_pred,
            hidden_nonlinearity=activation_function,
        )

    def forward(self, state: torch.Tensor, noised_action: torch.Tensor, sigma: torch.Tensor, extra_inputs: Dict) -> torch.Tensor:
        minibatch_size = state.shape[0]
        sigma = sigma.view(minibatch_size, -1)
        embedded_sigma = self.sigma_embedding(sigma)
        flat_noised_action = noised_action.view(minibatch_size, -1)
        flat_output = self.model(torch.cat([state, flat_noised_action, embedded_sigma], dim=-1))
        return flat_output.view(noised_action.shape)
