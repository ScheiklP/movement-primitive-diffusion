from pathlib import Path
import torch

from omegaconf import DictConfig
from typing import Dict, Union, Tuple

from movement_primitive_diffusion.agents.base_agent import BaseAgent


class BCAgent(BaseAgent):
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
    ):
        super().__init__(
            process_batch_config=process_batch_config,
            model_config=model_config,
            optimizer_config=optimizer_config,
            lr_scheduler_config=lr_scheduler_config,
            encoder_config=encoder_config,
            ema_config=ema_config,
            use_ema=use_ema,
            device=device,
        )

    def train_step(self, batch: Dict):
        """
        Executes a single training step on a mini-batch of data
        Args:
            batch: Mini batch dictionary

        Returns:
            Current train loss
        """
        # Process batch to get observation, action and extra inputs
        action, observation, extra_inputs = self.process_batch(batch)

        # Set model to train mode
        self.model.train()
        self.encoder.train()

        # Encoding observation
        state = self.encoder(observation)

        # Forward pass, calculate loss
        loss = self.model.loss(state, action, extra_inputs)

        # Backward pass
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.lr_scheduler.step()

        # Update EMA weights
        # NOTE: This does not change the model weights, but only the EMA weights. The EMA weights are only used during inference.
        if self.use_ema:
            self.ema_model.step(self.model.parameters())
            self.ema_encoder.step(self.encoder.parameters())

        return loss.item()

    @torch.no_grad()
    def predict(self, observation: Dict, extra_inputs: Dict) -> torch.Tensor:
        """Method for predicting one step with input data"""
        # Load the EMA weights
        if self.use_ema:
            self.ema_model.store(self.model.parameters())
            self.ema_encoder.store(self.encoder.parameters())
            self.ema_model.copy_to(self.model.parameters())
            self.ema_encoder.copy_to(self.encoder.parameters())

        # Set model to eval mode
        self.model.eval()
        self.encoder.eval()

        state = self.encoder(observation)

        action = self.model.forward(state, extra_inputs)

        # Load back the original weights
        if self.use_ema:
            self.ema_model.restore(self.model.parameters())
            self.ema_encoder.restore(self.encoder.parameters())

        return action

    @torch.no_grad()
    def evaluate(self, batch: Dict) -> Tuple[float, float, float]:
        """
        Method for evaluating the model on one batch of data
        Args:
            batch: Mini batch dictionary

        Returns:
            test loss
        """
        # Load the EMA weights
        if self.use_ema:
            self.ema_model.store(self.model.parameters())
            self.ema_encoder.store(self.encoder.parameters())
            self.ema_model.copy_to(self.model.parameters())
            self.ema_encoder.copy_to(self.encoder.parameters())

        # Set model to eval mode
        self.model.eval()
        self.encoder.eval()

        # Process batch to get observation, action and extra inputs
        action, observation, extra_inputs = self.process_batch(batch)

        with torch.no_grad():
            # Encoding observation
            state = self.encoder(observation)
            # NOTE: In contrast to the DiffusionAgent, we do not return the denoised/predicted action, but
            # the deviations, because the NLL model does not predict full trajectories, but pairs of points on the
            # trajectory so calculation of deviations differs from model to model.
            eval_loss, start_point_deviation, end_point_deviation = self.model.loss(state, action, extra_inputs, return_deviations=True)

        # Load back the original weights
        if self.use_ema:
            self.ema_model.restore(self.model.parameters())
            self.ema_encoder.restore(self.encoder.parameters())

        return eval_loss.item(), start_point_deviation.item(), end_point_deviation.item()

    def load_pretrained(self, path: Union[str, Path]):
        """
        Loads a pretrained model
        Args:
            path: Path to the pretrained model

        Returns:

        """
        state_dict = torch.load(path)
        self.model.load_state_dict(state_dict["model"])
        self.encoder.load_state_dict(state_dict["encoder"])
        if self.use_ema:
            self.ema_model.load_state_dict(self.model.state_dict())
            self.ema_encoder.load_state_dict(self.encoder.state_dict())

    def save_model(self, path: Union[str, Path]):
        """
        Saves the current model
        Args:
            path: Path to save the model

        Returns:

        """
        # TODO do we actually save model, or ema_model?
        # TODO do we save the optimizer and lr_scheduler to continue training?
        state_dict = {"model": self.model.state_dict(), "encoder": self.encoder.state_dict()}
        torch.save(state_dict, path)
