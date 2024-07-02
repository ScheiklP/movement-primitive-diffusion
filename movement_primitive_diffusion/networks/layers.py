import torch
import hydra
import numpy as np

from omegaconf import DictConfig
from torch import Tensor
from typing import List, Tuple


class PassThrough(torch.nn.Module):
    """Passes the input through unchanged."""

    def forward(self, x: Tensor) -> Tensor:
        return x

    def out_shape(self, in_shape: List[int]) -> List[int]:
        return in_shape


class FlattenTime(torch.nn.Module):
    """Flattens the input along the time dimension."""

    def forward(self, x: Tensor) -> Tensor:
        assert x.dim() == 3
        return x.view(x.size(0), -1)

    def out_shape(self, in_shape: List[int]) -> List[int]:
        return [in_shape[0], np.prod(in_shape[1:])]


class GaussianDistributionNetwork(torch.nn.Module):
    """A network that outputs the mean and variance of a Gaussian distribution.


    Args:
        network_config: The configuration of the network for instatiation with hydra.
        share_parameters: Whether to share the parameters of the mean and variance (one network which output's are split in half) or to use two separate networks.
        softplus_lower_bound: The lower bound of the softplus function to avoid numerical instabilities. This is added to the variance output of the network.

    Example:
        >>> network = GaussianDistributionNetwork({"_target_": "torch.nn.Linear", "in_features": 10, "out_features": 20}, share_parameters=True)
        >>> x = torch.randn(5, 10)
        >>> mean, variance = network(x)
        >>> mean.shape
        torch.Size([5, 10])
        >>> variance.shape
        torch.Size([5, 10])

        >>> network = GaussianDistributionNetwork({"_target_": "torch.nn.Linear", "in_features": 10, "out_features": 20}, share_parameters=False)
        >>> x = torch.randn(5, 10)
        >>> mean, variance = network(x)
        >>> mean.shape
        torch.Size([5, 20])
        >>> variance.shape
        torch.Size([5, 20])
    """

    def __init__(
        self,
        network_config: DictConfig,
        share_parameters: bool = False,
        softplus_lower_bound: float = 1e-2,
    ) -> None:
        super().__init__()

        if share_parameters:
            self.network = hydra.utils.instantiate(network_config)
            self.forward = self.forward_shared
        else:
            self.mean_network = hydra.utils.instantiate(network_config)
            self.variance_network = hydra.utils.instantiate(network_config)
            self.forward = self.forward_independent

        self.softplus = torch.nn.Softplus()
        self.softplus_lower_bound = softplus_lower_bound

    def forward_independent(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        mean = self.mean_network(x)
        variance = self.softplus(self.variance_network(x))

        variance = variance + self.softplus_lower_bound

        return mean, variance

    def forward_shared(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        mean_variance = self.network(x)
        assert mean_variance.shape[-1] % 2 == 0, "Last dimension must be even."

        mean = mean_variance[..., : mean_variance.shape[-1] // 2]
        variance = mean_variance[..., mean_variance.shape[-1] // 2 :]

        variance = variance + self.softplus_lower_bound

        return mean, variance
