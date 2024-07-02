import hydra
import torch

from omegaconf import DictConfig
from typing import Dict, List, Optional, Union

from movement_primitive_diffusion.networks.mlp import ResidualMLPBlock


class ParameterSpaceResidualMLPInnerModel(torch.nn.Module):
    def __init__(
        self,
        state_size: int,
        action_size: int,
        sigma_embedding_config: DictConfig,
        hidden_sizes: List[int] = [512, 256, 128],
        blocks_per_layer: int = 1,
        hidden_nonlinearity: str = "LeakyReLU",
        dropout_rate: float = 0.0,
        spectral_norm: bool = False,
        norm: Optional[str] = None,
    ):
        super().__init__()

        # Add a variable to hold the latest predicted parameters
        self.latest_parameters: Union[None, torch.Tensor] = None

        # Sigma embedding
        self.sigma_embedding: torch.nn.Module = hydra.utils.instantiate(sigma_embedding_config)

        # Size of encoded observation + size of one action times the number of time steps to predict + size of sigma embedding
        dummy_sigma = torch.zeros((1, 1))
        embedded_sigma = self.sigma_embedding(dummy_sigma)

        # size of encoded observation + size of one action times the number of time steps to predict + size of sigma embedding
        input_size = state_size + action_size + embedded_sigma.shape[-1]

        # Determine the input and output sizes of all residual blocks
        all_dims = [input_size] + list(hidden_sizes)
        in_out = list(zip(all_dims[:-1], all_dims[1:]))

        # Build a sequence of residual blocks
        sequence = []
        # The first linear layer changes the input size
        # The blocks keep the same input and output size
        for input_channels, output_channels in in_out:
            sequence.append(torch.nn.Linear(input_channels, output_channels))
            layer_sequence = []
            for _ in range(blocks_per_layer):
                layer_sequence.append(
                    ResidualMLPBlock(
                        num_neurons=output_channels,
                        hidden_nonlinearity=hidden_nonlinearity,
                        dropout_rate=dropout_rate,
                        spectral_norm=spectral_norm,
                        norm=norm,
                    )
                )
            sequence.extend(layer_sequence)

        # Add the final linear layer that outputs the parameters of the ProDMP
        sequence.append(torch.nn.Linear(all_dims[-1], action_size))

        self.model = torch.nn.Sequential(*sequence)

    def forward(self, state: torch.Tensor, noised_action: torch.Tensor, sigma: torch.Tensor, extra_inputs: Dict) -> torch.Tensor:
        minibatch_size = state.shape[0]
        sigma = sigma.view(minibatch_size, -1)
        embedded_sigma = self.sigma_embedding(sigma)
        params = self.model(torch.cat([state, noised_action, embedded_sigma], dim=-1))

        self.latest_parameters = params

        return params
