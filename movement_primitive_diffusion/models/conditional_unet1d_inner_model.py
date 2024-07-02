import torch
import hydra

from omegaconf import DictConfig
from typing import Dict, Optional, List

from movement_primitive_diffusion.networks.conv1d import Downsample1d, Upsample1d, Conv1dBlock, ConditionalResidualBlock1D


# Adapted from https://github.com/columbia-ai-robotics/diffusion_policy/blob/main/diffusion_policy/model/diffusion/conditional_unet1d.py
class ConditionalUnet1D(torch.nn.Module):
    def __init__(
        self,
        action_size: int,
        state_size: int,
        sigma_embedding_config: DictConfig,
        local_condition_size: Optional[int] = None,
        down_sizes: List[int] = [512, 1024, 2048],
        kernel_size: int = 5,
        n_groups: int = 8,
        predict_scale_of_condition: bool = True,
    ):
        super().__init__()
        all_dims = [action_size] + list(down_sizes)
        start_dim = down_sizes[0]

        self.sigma_embedding: torch.nn.Module = hydra.utils.instantiate(sigma_embedding_config)

        condition_size = sigma_embedding_config.embedding_size
        if state_size is not None:
            condition_size += state_size

        in_out = list(zip(all_dims[:-1], all_dims[1:]))

        local_condition_encoder = None
        if local_condition_size is not None:
            _, output_channels = in_out[0]
            input_channels = local_condition_size
            local_condition_encoder = torch.nn.ModuleList(
                [
                    # local condition encoder for down encoder
                    ConditionalResidualBlock1D(
                        input_channels,
                        output_channels,
                        condition_size=condition_size,
                        kernel_size=kernel_size,
                        n_groups=n_groups,
                        predict_scale_of_condition=predict_scale_of_condition,
                    ),
                    # local condition encoder for up encoder
                    ConditionalResidualBlock1D(
                        input_channels,
                        output_channels,
                        condition_size=condition_size,
                        kernel_size=kernel_size,
                        n_groups=n_groups,
                        predict_scale_of_condition=predict_scale_of_condition,
                    ),
                ]
            )

        latent_size = all_dims[-1]
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

        up_modules = torch.nn.ModuleList([])
        for ind, (input_channels, output_channels) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(
                torch.nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            output_channels * 2,
                            input_channels,
                            condition_size=condition_size,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                            predict_scale_of_condition=predict_scale_of_condition,
                        ),
                        ConditionalResidualBlock1D(
                            input_channels,
                            input_channels,
                            condition_size=condition_size,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                            predict_scale_of_condition=predict_scale_of_condition,
                        ),
                        Upsample1d(input_channels) if not is_last else torch.nn.Identity(),
                    ]
                )
            )

        final_conv = torch.nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            torch.nn.Conv1d(start_dim, action_size, 1),
        )

        self.local_condition_encoder = local_condition_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

    def forward(
        self,
        state: torch.Tensor,
        noised_action: torch.Tensor,
        sigma: torch.Tensor,
        extra_inputs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Forward pass of the model.

        Args:

            state (torch.Tensor): State observation as condition for the model.
            noised_action (torch.Tensor): Noised action as input to the model.
            sigma (torch.Tensor): Sigma as condition for the model.
            extra_inputs (Dict[str, torch.Tensor]): Dict with key "local_condition" and tensor value (B,T,local_condition_size).

        Returns:
            torch.Tensor: Model output. Shape (B, t_pred, action_size).
        """
        # swap from (B, T, H) to (B, H, T) to convolve over time, not features
        noised_action = noised_action.swapaxes(-1, -2)
        minibatch_size = noised_action.shape[0]

        # Embed sigma
        global_feature = self.sigma_embedding(sigma.view(minibatch_size, -1))

        global_feature = torch.cat([global_feature, state], axis=-1)

        # encode local features
        h_local = list()
        if extra_inputs is not None and "local_condition" in extra_inputs:
            local_condition = extra_inputs["local_condition"]
            local_condition = local_condition.swapaxes(-1, -2)
            resnet, resnet2 = self.local_condition_encoder
            x = resnet(local_condition, global_feature)
            h_local.append(x)
            x = resnet2(local_condition, global_feature)
            h_local.append(x)

        x = noised_action
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            if idx == 0 and len(h_local) > 0:
                x = x + h_local[0]
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            if idx == len(self.up_modules) and len(h_local) > 0:
                x = x + h_local[1]
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)

        # swap back to (B, T, H)
        denoised_action = x.swapaxes(-1, -2)

        return denoised_action
