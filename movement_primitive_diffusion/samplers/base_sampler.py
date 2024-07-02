from abc import ABC, abstractmethod
from typing import Dict, Optional, Callable

import torch


class BaseSampler(ABC):
    """
    Base class for samplers
    """

    @abstractmethod
    def __init__(self, device="cpu", dtype=torch.float32):
        self.device = device
        self.dtype = dtype

    @abstractmethod
    def sample(
        self,
        model: torch.nn.Module,
        state: torch.Tensor,
        action: torch.Tensor,
        sigmas: torch.Tensor,
        extra_inputs: Optional[Dict] = None,
        callback: Optional[Callable] = None,
    ) -> torch.Tensor:
        """
        Sample loop to calculate the noise free action

        Args:
            model: Model to be used for sampling
            state: State tensor of observations
            action: Initial noisy action
            sigmas: Sigma values to iterate over during sampling
            extra_inputs: Extra inputs to the model
            callback: Callback function to be called after each denoising step

        Returns:
            Denoised action
        """
        raise NotImplementedError
