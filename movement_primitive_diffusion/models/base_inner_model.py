from abc import ABC, abstractmethod
from typing import Dict
import torch


class BaseInnerModel(torch.nn.Module, ABC):
    @abstractmethod
    def forward(self, state: torch.Tensor, noised_action: torch.Tensor, sigma: torch.Tensor, extra_inputs: Dict) -> torch.Tensor:
        raise NotImplementedError
