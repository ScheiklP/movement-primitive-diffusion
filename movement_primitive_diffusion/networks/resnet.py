import torch
from typing import List, Callable
from robomimic.models.obs_core import VisualCore


def replace_submodules(
    root_module: torch.nn.Module,
    predicate: Callable[[torch.nn.Module], bool],
    func: Callable[[torch.nn.Module], torch.nn.Module],
) -> torch.nn.Module:
    """
    From https://github.com/columbia-ai-robotics/diffusion_policy/blob/main/diffusion_policy/common/pytorch_util.py#L43
    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)

    bn_list = [k.split(".") for k, m in root_module.named_modules(remove_duplicate=True) if predicate(m)]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule(".".join(parent))
        if isinstance(parent_module, torch.nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, torch.nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all BN are replaced
    bn_list = [k.split(".") for k, m in root_module.named_modules(remove_duplicate=True) if predicate(m)]
    assert len(bn_list) == 0
    return root_module


def get_resnet(input_shape: List[int], output_size: int):
    """Get ResNet model from torchvision.models

    Args:
        input_shape: Shape of input image (C, H, W).
        output_size: Size of output feature vector.
    """

    resnet = VisualCore(
        input_shape=input_shape,
        backbone_class="ResNet18Conv",
        backbone_kwargs=dict(
            input_coord_conv=False,
            pretrained=False,
        ),
        pool_class="SpatialSoftmax",
        pool_kwargs=dict(
            num_kp=32,
            learnable_temperature=False,
            temperature=1.0,
            noise_std=0.0,
            output_variance=False,
        ),
        flatten=True,
        feature_dimension=output_size,
    )

    replace_submodules(
        root_module=resnet,
        predicate=lambda x: isinstance(x, torch.nn.BatchNorm2d),
        func=lambda x: torch.nn.GroupNorm(num_groups=x.num_features // 16, num_channels=x.num_features),
    )
    return resnet


class IndependentTimeStepsResNet(torch.nn.Module):
    def __init__(
        self,
        feature_size: List[int],
        output_size: int,
        flatten_time: bool = True,
    ) -> None:
        super().__init__()

        self.resnet: VisualCore = get_resnet(feature_size, output_size)
        self.flatten_time = flatten_time

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        assert x.dim() == 5

        # Move time dimension into batch dimension to process time steps individually (B, T, C, H, W) -> (B * T, C, H, W)
        original_shape = x.shape
        x = x.reshape(original_shape[0] * original_shape[1], *original_shape[2:])

        # Forward pass (B * T, C, H, W) -> (B * T, out_channels)
        output = self.resnet(x)

        if self.flatten_time:
            # Reshape output to (B, T*out_channels)
            output = output.view(original_shape[0], original_shape[1] * output.size(1))
        else:
            # Reshape output to (B, T, out_channels)
            output = output.view(original_shape[0], original_shape[1], output.size(1))

        return output


class DependentTimeStepsResNet(torch.nn.Module):
    def __init__(
        self,
        feature_size: List[int],
        output_size: int,
        time_steps: int,
    ) -> None:
        super().__init__()

        assert len(feature_size) == 3, "Feature size must be (C, H, W)"

        # Multiply number of channels by time steps to process time steps together
        feature_size[0] *= time_steps

        self.resnet: VisualCore = get_resnet(feature_size, output_size)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        assert x.dim() == 5

        # Move time dimension into channel dimension to process time steps together (B, T, C, H, W) -> (B, T*C, H, W)
        original_shape = x.shape
        x = x.reshape(original_shape[0], original_shape[1] * original_shape[2], *original_shape[3:])

        # Forward pass (B, T*C, H, W) -> (B, out_channels)
        output = self.resnet(x)

        return output
