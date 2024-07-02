import torch


# Adapted from https://github.com/columbia-ai-robotics/diffusion_policy/blob/main/diffusion_policy/model/diffusion/conv1d_components.py
class Downsample1d(torch.nn.Module):
    def __init__(self, in_channels: int, kernel_size: int = 3, stride: int = 2, padding: int = 1):
        super().__init__()
        self.conv = torch.nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# Adapted from https://github.com/columbia-ai-robotics/diffusion_policy/blob/main/diffusion_policy/model/diffusion/conv1d_components.py
class Upsample1d(torch.nn.Module):
    def __init__(self, in_channels: int, kernel_size: int = 4, stride: int = 2, padding: int = 1):
        super().__init__()
        self.conv = torch.nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# Adapted from https://github.com/columbia-ai-robotics/diffusion_policy/blob/main/diffusion_policy/model/diffusion/conv1d_components.py
class Conv1dBlock(torch.nn.Module):
    """
    Conv1d --> GroupNorm --> Mish
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, n_groups=8):
        super().__init__()

        self.block = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            torch.nn.GroupNorm(n_groups, out_channels),
            torch.nn.Mish(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# Adapted from https://github.com/columbia-ai-robotics/diffusion_policy/blob/main/diffusion_policy/model/diffusion/conditional_unet1d.py
class ConditionalResidualBlock1D(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        condition_size: int,
        kernel_size: int = 3,
        n_groups: int = 8,
        predict_scale_of_condition: bool = False,
    ):
        super().__init__()

        self.blocks = torch.nn.ModuleList(
            [
                Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
                Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
            ]
        )

        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias
        cond_channels = out_channels
        if predict_scale_of_condition:
            cond_channels = out_channels * 2
        self.predict_scale_of_condition = predict_scale_of_condition
        self.out_channels = out_channels
        self.cond_encoder = torch.nn.Sequential(
            torch.nn.Mish(),
            torch.nn.Linear(condition_size, cond_channels),
        )

        # make sure dimensions compatible
        self.residual_conv = torch.nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else torch.nn.Identity()

    def forward(self, x, cond):
        """
        x : [ batch_size x in_channels x horizon ]
        cond : [ batch_size x condition_size]

        returns:
        out : [ batch_size x out_channels x horizon ]
        """
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)
        embed = embed.unsqueeze(-1)
        if self.predict_scale_of_condition:
            # Split output of cond_encoder into scale and bias
            embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1)
            scale = embed[:, 0, ...]
            bias = embed[:, 1, ...]
            out = scale * out + bias
        else:
            out = out + embed
        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out
