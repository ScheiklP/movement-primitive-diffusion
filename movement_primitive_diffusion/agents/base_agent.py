import hydra
import torch

from abc import ABC, abstractmethod
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from typing import Dict, Union, Optional

from diffusers.training_utils import EMAModel

from movement_primitive_diffusion.datasets.process_batch import ProcessBatch
from movement_primitive_diffusion.encoder import Encoder
from movement_primitive_diffusion.models.base_model import BaseModel


class BaseAgent(ABC):
    def __init__(
        self,
        model_config: DictConfig,
        optimizer_config: DictConfig,
        lr_scheduler_config: DictConfig,
        encoder_config: DictConfig,
        process_batch_config: DictConfig,
        ema_config: DictConfig,
        use_ema: bool = False,
        device: Union[str, torch.device] = "cpu",
        special_optimizer_function: bool = False,
        special_optimizer_config: Optional[DictConfig] = None,
    ):
        super().__init__()

        self.process_batch: ProcessBatch = hydra.utils.instantiate(process_batch_config)
        self.encoder: Encoder = hydra.utils.instantiate(encoder_config)

        if hasattr(model_config, "inner_model_config"):
            if hasattr(model_config.inner_model_config, "action_size"):
                model_config.inner_model_config.action_size = self.process_batch.action_size
            if hasattr(model_config.inner_model_config, "state_size"):
                model_config.inner_model_config.state_size = self.encoder.output_size

        self.model: BaseModel = hydra.utils.instantiate(model_config)

        if special_optimizer_function:
            # Transformer based inner models (diffusion policy transformer and beso) handle setting weight decay for their
            # parameter groups manually.
            if hasattr(self.model, "inner_model") and hasattr(self.model.inner_model, "configure_optimizers"):
                # TODO: it is not very nice that we force these kwargs here.
                # Maybe we could use inspect to get the signature of the function and match against special_optimizer_config
                self.optimizer = self.model.inner_model.configure_optimizers(
                    learning_rate=special_optimizer_config.learning_rate,
                    weight_decay=special_optimizer_config.model_weight_decay,
                    betas=special_optimizer_config.betas,
                    eps=special_optimizer_config.eps,
                )
                encoder_param_group = {
                    "params": list(self.encoder.parameters()),
                    "weight_decay": special_optimizer_config.encoder_weight_decay,
                }
                self.optimizer.add_param_group(encoder_param_group)
            else:
                raise ValueError(f"You set special_optimizer_function to True, but the model {self.model} does not have a configure_optimizers function.")
        else:
            self.optimizer: torch.optim.Optimizer = hydra.utils.instantiate(optimizer_config, params=list(self.model.parameters()) + list(self.encoder.parameters()))

        self.lr_scheduler: torch.optim.lr_scheduler.LRScheduler = hydra.utils.call(lr_scheduler_config, optimizer=self.optimizer)

        # Move all networks to correct device
        self.device = torch.device(device)
        self.encoder.to(device)
        self.model.to(device)

        # Exponential moving average of the model and encoder parameters
        self.use_ema = use_ema
        # TODO model and encoder can probably just put into one EMAModel
        self.ema_model = None
        self.ema_encoder = None
        if use_ema:
            ema_config = OmegaConf.to_container(ema_config, resolve=True)
            self.ema_model = EMAModel(parameters=self.model.parameters(), model_cls=self.model.__class__, model_config=model_config, **ema_config)
            self.ema_encoder = EMAModel(self.encoder.parameters(), model_cls=self.encoder.__class__, model_config=encoder_config, **ema_config)
            self.ema_model.to(device)
            self.ema_encoder.to(device)

    @abstractmethod
    def train_step(self, batch: Dict):
        """
        Executes a single training step on a mini-batch of data
        Args:
            batch: Mini batch dictionary

        Returns:
            Current train loss
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, observation: Dict, extra_inputs: Dict) -> torch.Tensor:
        """Method for predicting one step with input data"""
        raise NotImplementedError

    def use_ema_weights(self) -> None:
        """Method for using the EMA weights during inference"""
        if self.ema_model is None or self.ema_encoder is None:
            raise RuntimeWarning("EMA model or EMA encoder is None.")
        elif not self.use_ema:
            raise RuntimeWarning("EMA weights are already in use.")
        else:
            self.use_ema = False  # -> skip if self.use_ema in predict method
            # Store current weights in EMA model
            self.ema_model.store(self.model.parameters())
            self.ema_encoder.store(self.encoder.parameters())
            # Copy EMA weights to model
            self.ema_model.copy_to(self.model.parameters())
            self.ema_encoder.copy_to(self.encoder.parameters())

    def restore_model_weights(self) -> None:
        if self.ema_model is None or self.ema_encoder is None:
            raise RuntimeWarning("EMA model or EMA encoder is None.")
        elif self.use_ema:
            raise RuntimeWarning("Model weights are already restored.")
        else:
            self.use_ema = True  # -> do not skip if self.use_ema in predict method
            # Restore model weights from EMA model
            self.ema_model.restore(self.model.parameters())
            self.ema_encoder.restore(self.encoder.parameters())

    def predict_upsampled(self, observation: Dict, extra_inputs: Dict, input_dt: float, output_dt: float, mode: str = "linear") -> torch.Tensor:
        """Method for predicting with a higher resolution."""

        # Denoised action has shape (batch_size, t_pred, action_dim)
        denoised_action = self.predict(observation, extra_inputs)

        # Calculate the ratio between the input and output dt used for upsampling
        dt_ratio = input_dt / output_dt

        # Interpolate wants a tensor of shape (batch_size, action_dim, t_pred) to interpolate over the time dimension -> permute
        upsampled_action = torch.nn.functional.interpolate(denoised_action.permute(0, 2, 1), scale_factor=dt_ratio, mode=mode, align_corners=True).permute(0, 2, 1)

        return upsampled_action

    @abstractmethod
    def evaluate(self, batch: Dict):
        """
        Method for evaluating the model on one batch of data
        Args:
            batch: Mini batch dictionary

        Returns:
            test loss
        """
        raise NotImplementedError

    @abstractmethod
    def load_pretrained(self, path: Union[str, Path]):
        """
        Loads a pretrained model
        Args:
            path: Path to the pretrained model

        Returns:

        """
        raise NotImplementedError

    @abstractmethod
    def save_model(self, path: Union[str, Path]):
        """
        Saves the current model
        Args:
            path: Path to save the model

        Returns:

        """
        raise NotImplementedError
