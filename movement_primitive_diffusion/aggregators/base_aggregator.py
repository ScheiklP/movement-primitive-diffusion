import torch
from abc import ABC, abstractmethod
from typing import Dict


class Aggregator(ABC):
    @abstractmethod
    def __call__(self, encodings: Dict[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError
