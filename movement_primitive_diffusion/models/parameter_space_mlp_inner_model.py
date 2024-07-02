import torch
import hydra

from omegaconf import DictConfig
from typing import Dict, Union

from movement_primitive_diffusion.networks.mlp import SimpleMLP
from movement_primitive_diffusion.models.base_inner_model import BaseInnerModel


class ParameterSpaceMLPInnerModel(BaseInnerModel):
    def __init__(
        self,
        state_size: int,
        action_size: int,
        num_layers: int,
        num_neurons: int,
        sigma_embedding_config: DictConfig,
        hidden_nonlinearity: Union[str, torch.nn.Module] = "LeakyReLU",
    ) -> None:
        super().__init__()

        self.sigma_embedding: torch.nn.Module = hydra.utils.instantiate(sigma_embedding_config)
        dummy_sigma = torch.zeros((1, 1))
        embedded_sigma = self.sigma_embedding(dummy_sigma)

        # size of encoded observation + size of one action times the number of time steps to predict + size of sigma embedding
        input_size = state_size + action_size + embedded_sigma.shape[-1]

        self.model = SimpleMLP(
            input_size=input_size,
            output_size=action_size,
            num_layers=num_layers,
            num_neurons=num_neurons,
            hidden_nonlinearity=hidden_nonlinearity,
        )

    def forward(self, state: torch.Tensor, noised_action: torch.Tensor, sigma: torch.Tensor, extra_inputs: Dict) -> torch.Tensor:
        minibatch_size = state.shape[0]
        sigma = sigma.view(minibatch_size, -1)
        embedded_sigma = self.sigma_embedding(sigma)
        return self.model(torch.cat([state, noised_action, embedded_sigma], dim=-1))
