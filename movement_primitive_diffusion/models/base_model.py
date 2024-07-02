import torch

from abc import ABC, abstractmethod
from typing import Dict, Tuple, Union


class BaseModel(torch.nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def loss(self, state: torch.Tensor, action: torch.Tensor, sigma: torch.Tensor, extra_inputs: Dict, return_denoised: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        raise NotImplementedError

    @abstractmethod
    def forward(self, state: torch.Tensor, noised_action: torch.Tensor, sigma: torch.Tensor, extra_inputs: Dict) -> torch.Tensor:
        raise NotImplementedError
