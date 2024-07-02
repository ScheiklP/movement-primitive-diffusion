import torch
import hydra

from omegaconf import DictConfig
from typing import Dict, List, Union

from movement_primitive_diffusion.networks.conv1d import Downsample1d, Conv1dBlock, ConditionalResidualBlock1D
from movement_primitive_diffusion.utils.mp_utils import ProDMPHandler


class ProDMPResNet1DInnerModel(torch.nn.Module):
    def __init__(
        self,
        action_size: int,
        state_size: int,
        sigma_embedding_config: DictConfig,
        prodmp_handler_config: DictConfig,
        down_sizes: List[int] = [512, 1024, 2048],
        kernel_size: int = 5,
        n_groups: int = 8,
        predict_scale_of_condition: bool = True,
    ):
        super().__init__()

        # Initialize ProDMPHandler
        prodmp_handler_config.num_dof = action_size
        self.prodmp_handler: ProDMPHandler = hydra.utils.instantiate(prodmp_handler_config)

        # Add a variable to hold the latest predicted ProDMP parameters
        self.latest_prodmp_parameters: Union[None, torch.Tensor] = None

        # Initialize sigma embedding
        self.sigma_embedding: torch.nn.Module = hydra.utils.instantiate(sigma_embedding_config)

        # Size of condition vector is the size of the sigma embedding and the state observation
        condition_size = sigma_embedding_config.embedding_size + state_size

        # Determine the input and output sizes of all residual blocks
        all_dims = [action_size] + list(down_sizes)
        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        latent_size = all_dims[-1]

        # Initialize the middle modules with constant latent size
        self.mid_modules = torch.nn.ModuleList(
            [
                ConditionalResidualBlock1D(
                    latent_size,
                    latent_size,
                    condition_size=condition_size,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                    predict_scale_of_condition=predict_scale_of_condition,
                ),
                ConditionalResidualBlock1D(
                    latent_size,
                    latent_size,
                    condition_size=condition_size,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                    predict_scale_of_condition=predict_scale_of_condition,
                ),
            ]
        )

        # Initialize the down modules
        down_modules = torch.nn.ModuleList([])
        for ind, (input_channels, output_channels) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(
                torch.nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            input_channels,
                            output_channels,
                            condition_size=condition_size,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                            predict_scale_of_condition=predict_scale_of_condition,
                        ),
                        ConditionalResidualBlock1D(
                            output_channels,
                            output_channels,
                            condition_size=condition_size,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                            predict_scale_of_condition=predict_scale_of_condition,
                        ),
                        Downsample1d(output_channels) if not is_last else torch.nn.Identity(),
                    ]
                )
            )
        self.down_modules = down_modules

        # Final convolution from size of middle modules to prodmp encoding
        final_conv = torch.nn.Sequential(
            Conv1dBlock(latent_size, latent_size, kernel_size=kernel_size),
            # NOTE: The choice of kernel_size, stride, and padding halves the size in every Downsample1d and this last conv
            torch.nn.Conv1d(latent_size, self.prodmp_handler.encoding_size, kernel_size=3, stride=2, padding=1),
        )
        self.final_conv = final_conv

    def forward(
        self,
        state: torch.Tensor,
        noised_action: torch.Tensor,
        sigma: torch.Tensor,
        extra_inputs: Dict[str, torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass of the model.

        Args:

            state (torch.Tensor): State observation as condition for the model.
            noised_action (torch.Tensor): Noised action as input to the model.
            sigma (torch.Tensor): Sigma as condition for the model.
            extra_inputs (Dict[str, torch.Tensor]): Dict with key initial values for position and velocity.

        Returns:
            torch.Tensor: Model output. Shape (B, t_pred, action_size).
        """
        # swap from (B, T, H) to (B, H, T) to convolve over time, not features
        noised_action = noised_action.swapaxes(-1, -2)
        minibatch_size = noised_action.shape[0]

        # Embed sigma
        global_feature = self.sigma_embedding(sigma.view(minibatch_size, -1))

        global_feature = torch.cat([global_feature, state], axis=-1)

        x = noised_action
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        x = self.final_conv(x).squeeze(-1)

        # Set the latest prodmp parameters so that they can be used in predict_upsampled
        self.latest_prodmp_parameters = x

        trajectory = self.prodmp_handler.decode(x, **extra_inputs)

        return trajectory
