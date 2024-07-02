import torch

from abc import ABC, abstractmethod
from torch.utils.data import DataLoader
from typing import Tuple, Optional

from movement_primitive_diffusion.agents.base_agent import BaseAgent


class Scaling(ABC):
    @abstractmethod
    def __call__(self, sigma: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns c_skip, c_out, c_in, c_noise scaling factors for the diffusion model"""
        raise NotImplementedError


class KarrasScaling(Scaling, torch.nn.Module):
    def __init__(self, sigma_data: Optional[float] = 0.5) -> None:
        super().__init__()
        self.sigma_data = sigma_data

    def set_sigma_data(self, sigma_data: float) -> None:
        self.sigma_data = sigma_data

    def __call__(self, sigma: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.sigma_data is None:
            raise RuntimeError("Sigma data must be set before calling KarrasScaling")
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2) ** 0.5
        c_in = 1.0 / (sigma**2 + self.sigma_data**2) ** 0.5
        c_noise = torch.log(sigma) / 4.0

        return c_skip, c_out, c_in, c_noise

    def calculate_sigma_data_of_action(self, agent: BaseAgent, dataloader: DataLoader) -> float:
        """Computes the standard deviation of the action data."""
        actions = []
        for batch in dataloader:
            action, _, _ = agent.process_batch(batch)
            actions.append(action)

        # Reshape actions to (batch_size * seq_len, action_size)
        actions = torch.cat(actions, dim=0)
        actions = actions.reshape((-1, actions.shape[-1]))

        # Compute standard deviation
        return actions.std().cpu().item()


class TrajectoryKarrasScaling(KarrasScaling):
    def __call__(self, sigma: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.sigma_data is None:
            raise RuntimeError("Sigma data must be set before calling KarrasScaling")
        c_skip = torch.zeros_like(sigma)
        c_out = torch.ones_like(sigma)
        c_in = 1.0 / (sigma**2 + self.sigma_data**2).sqrt()
        c_noise = torch.log(sigma) / 4.0

        return c_skip, c_out, c_in, c_noise


class Absolute(Scaling):
    def __call__(self, sigma: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        c_skip = torch.zeros_like(sigma)
        c_out = torch.ones_like(sigma)
        c_in = torch.ones_like(sigma)
        c_noise = sigma

        return c_skip, c_out, c_in, c_noise


class Epsilon(Scaling):
    def __call__(self, sigma: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        c_skip = torch.ones_like(sigma)
        c_out = torch.ones_like(sigma)
        c_in = torch.ones_like(sigma)
        c_noise = sigma

        return c_skip, c_out, c_in, c_noise
