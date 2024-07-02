import hydra
import torch

from omegaconf import DictConfig
from pathlib import Path
from typing import Dict, Tuple, Union, Optional
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from movement_primitive_diffusion.agents.base_agent import BaseAgent


class DiscreteTimeDiffusionAgent(BaseAgent):
    def __init__(
        self,
        model_config: DictConfig,
        optimizer_config: DictConfig,
        lr_scheduler_config: DictConfig,
        encoder_config: DictConfig,
        noise_scheduler_config: DictConfig,
        process_batch_config: DictConfig,
        t_obs: int,
        predict_past: bool,
        ema_config: DictConfig,
        use_ema: bool = False,
        device: Union[str, torch.device] = "cpu",
        num_inference_steps: Optional[int] = None,
        special_optimizer_function: bool = False,
        special_optimizer_config: Optional[DictConfig] = None,
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
            special_optimizer_function=special_optimizer_function,
            special_optimizer_config=special_optimizer_config,
        )

        self.t_obs = t_obs
        self.predict_past = predict_past

        # These are the sigmas used during inference (integrating the differential equation)
        self.noise_scheduler: DDPMScheduler = hydra.utils.instantiate(noise_scheduler_config)

        # Number of steps to integrate the differential equation during inference
        # If None, use the same value as num_train_timesteps
        if num_inference_steps is None:
            num_inference_steps = self.noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

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

        # Sample noise from a unit Gaussian
        noise = torch.randn_like(action, device=self.device)

        # Sample random time steps from 0 to num_train_timesteps for each batch element
        batch_size = action.shape[0]
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (batch_size,), device=self.device)
        for _ in range(action.ndim - timesteps.ndim):
            timesteps = timesteps.unsqueeze(-1)

        # Add noise to the action
        noised_action = self.noise_scheduler.add_noise(action, noise, timesteps)

        # Forward pass
        model_output = self.model(state, noised_action, timesteps, extra_inputs)

        # Compute loss
        prediction_type = self.noise_scheduler.config.prediction_type
        if prediction_type == "epsilon":
            target = noise
        elif prediction_type == "sample":
            target = action
        else:
            raise ValueError(f"Unknown prediction type {prediction_type}")
        loss = (model_output - target).pow(2).flatten().mean()

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

        # Encode the observation
        state = self.encoder(observation)

        # Sample random noisy action of shape (batch_size, ...) from a unit Gaussian
        batch_size = observation[list(observation.keys())[0]].shape[0]
        action = torch.randn((batch_size, *self.process_batch.action_shape), device=self.device)

        # Set the scheduler's time steps to inference
        self.noise_scheduler.set_timesteps(num_inference_steps=self.num_inference_steps, device=self.device)

        # Iteratively denoise the action
        for timestep in self.noise_scheduler.timesteps:
            timesteps = torch.full((batch_size,), timestep, device=self.device)
            for _ in range(action.ndim - timesteps.ndim):
                timesteps = timesteps.unsqueeze(-1)
            model_output = self.model(state, action, timesteps, extra_inputs)
            action = self.noise_scheduler.step(model_output, timestep, action).prev_sample

        # Load back the original weights
        if self.use_ema:
            self.ema_model.restore(self.model.parameters())
            self.ema_encoder.restore(self.encoder.parameters())

        if self.predict_past:
            return action[:, self.t_obs-1:, :]

        return action

    @torch.no_grad()
    def evaluate(self, batch: Dict) -> Tuple[float, float, float]:
        """
        Method for evaluating the model on one batch of data
        Args:
            batch: Mini batch dictionary

        Returns:
            test loss, start point deviation, end point deviation
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

            # Sample noise from a unit Gaussian
            noise = torch.randn_like(action, device=self.device)

            # Sample random time steps from 0 to num_train_timesteps for each batch element
            batch_size = action.shape[0]
            timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (batch_size,), device=self.device)
            for _ in range(action.ndim - timesteps.ndim):
                timesteps = timesteps.unsqueeze(-1)

            # Add noise to the action
            noised_action = self.noise_scheduler.add_noise(action, noise, timesteps)

            # Forward pass
            model_output = self.model(state, noised_action, timesteps, extra_inputs)

            # Compute loss
            prediction_type = self.noise_scheduler.config.prediction_type
            if prediction_type == "epsilon":
                target = noise
                denoised_action = action - model_output
            elif prediction_type == "sample":
                target = action
                denoised_action = model_output
            else:
                raise ValueError(f"Unknown prediction type {prediction_type}")

            eval_loss = (model_output - target).pow(2).flatten().mean()

            # Calculate the L2 error of the first and last action of the sequence
            start_point_deviation = torch.linalg.norm(action[:, 0, :] - denoised_action[:, 0, :], dim=-1).mean()
            end_point_deviation = torch.linalg.norm(action[:, -1, :] - denoised_action[:, -1, :], dim=-1).mean()

        # Load back the original weights
        if self.use_ema:
            self.ema_model.restore(self.model.parameters())
            self.ema_encoder.restore(self.encoder.parameters())

        return eval_loss.item(), start_point_deviation.item(), end_point_deviation.item()

    def load_pretrained(self, path: Union[str, Path]):
        """Loads a pretrained model

        Args:
            path: Path to the pretrained model

        Returns:

        """
        state_dict = torch.load(path)
        self.model.load_state_dict(state_dict["model"])
        self.encoder.load_state_dict(state_dict["encoder"])
        if self.use_ema:
            self.ema_model.load_state_dict(state_dict["ema_model"])
            self.ema_encoder.load_state_dict(state_dict["ema_encoder"])

        if "optimizer" in state_dict:
            self.optimizer.load_state_dict(state_dict["optimizer"])
        if "lr_scheduler" in state_dict:
            self.lr_scheduler.load_state_dict(state_dict["lr_scheduler"])

    def save_model(self, path: Union[str, Path], save_optimizer: bool = False, save_lr_scheduler: bool = False):
        """Saves the current model

        Args:
            path: Path to save the model

        Returns:

        """
        # Save state dicts of model and encoder
        state_dict = {"model": self.model.state_dict(), "encoder": self.encoder.state_dict()}

        # Save EMA state dicts if EMA is used
        if self.use_ema:
            state_dict["ema_model"] = self.ema_model.state_dict()
            state_dict["ema_encoder"] = self.ema_encoder.state_dict()

        # To be able to continue training, save optimizer and lr_scheduler state dicts
        if save_optimizer:
            state_dict["optimizer"] = self.optimizer.state_dict()
        if save_lr_scheduler:
            state_dict["lr_scheduler"] = self.lr_scheduler.state_dict()

        torch.save(state_dict, path)
