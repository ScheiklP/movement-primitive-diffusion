from pathlib import Path
import hydra
import torch

from omegaconf import DictConfig
from typing import Dict, Tuple, Union, Optional

from movement_primitive_diffusion.agents.base_agent import BaseAgent
from movement_primitive_diffusion.distributions import Distribution
from movement_primitive_diffusion.noise_schedulers import NoiseScheduler
from movement_primitive_diffusion.samplers.base_sampler import BaseSampler


class DiffusionAgent(BaseAgent):
    def __init__(
        self,
        model_config: DictConfig,
        optimizer_config: DictConfig,
        lr_scheduler_config: DictConfig,
        encoder_config: DictConfig,
        sigma_distribution_config: DictConfig,
        sampler_config: DictConfig,
        noise_scheduler_config: DictConfig,
        process_batch_config: DictConfig,
        t_obs: int,
        predict_past: bool,
        diffusion_steps: int,
        sigma_min: float,
        sigma_max: float,
        ema_config: DictConfig,
        use_ema: bool = False,
        device: Union[str, torch.device] = "cpu",
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

        # These are the sigmas used during denoising score matching to train the model on data
        self.sigma_distribution: Distribution = hydra.utils.instantiate(sigma_distribution_config)

        # This is the sampler used during inference
        self.sampler: BaseSampler = hydra.utils.instantiate(sampler_config)

        # These are the sigmas used during inference (integrating the differential equation)
        self.noise_scheduler: NoiseScheduler = hydra.utils.instantiate(noise_scheduler_config)

        # Number of steps to integrate the differential equation
        self.diffusion_steps = diffusion_steps

        # Min and max values for the sigmas used during inference
        # Note: In the config, we set the sigma_min and sigma_max values for sigma_distribution and noise_scheduler to be the same
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

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

        # Sampling sigma for the noise level used in the model
        # One sigma per batch element -> shape = (batch_size, 1)
        sigma = self.sigma_distribution.sample(shape=action.shape[0]).to(self.device)

        # Forward pass, calculate loss
        loss = self.model.loss(state, action, sigma, extra_inputs)

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

        batch_size = observation[list(observation.keys())[0]].shape[0]

        state = self.encoder(observation)

        sigmas = self.noise_scheduler.get_sigmas(self.diffusion_steps).to(self.device)

        # Sample random noisy action of shape (batch_size, ...) and scale it by sigma_max
        noised_action = torch.randn((batch_size, *self.process_batch.action_shape), device=self.device) * self.sigma_max

        denoised_action = self.sampler.sample(model=self.model, state=state, action=noised_action, sigmas=sigmas, extra_inputs=extra_inputs)

        # Load back the original weights
        if self.use_ema:
            self.ema_model.restore(self.model.parameters())
            self.ema_encoder.restore(self.encoder.parameters())

        if self.predict_past:
            return denoised_action[:, self.t_obs-1:, :]

        return denoised_action

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

        # Sampling sigma for the noise level used in the model
        # One sigma per batch element -> shape = (batch_size, 1)
        sigma = self.sigma_distribution.sample(shape=action.shape[0]).to(self.device)

        with torch.no_grad():
            # Encoding observation
            state = self.encoder(observation)
            eval_loss, denoised_action = self.model.loss(state, action, sigma, extra_inputs, return_denoised=True)

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


class ProDMPDiffusionAgent(DiffusionAgent):
    def predict_upsampled(self, observation: Dict, extra_inputs: Dict, input_dt: float, output_dt: float, mode: str = "linear") -> torch.Tensor:
        """Method for predicting with a higher resolution.

        In contrast to non-prodmp models, we do not interpolate the denoised action but decode the motion primitive directly for the upsampled time steps.
        """

        # Run denoising to generate denoised prodmp parameters in the inner model -> output of predict is not used
        denoised_action = self.predict(observation, extra_inputs)

        # Calculate the upsampled prediction times
        t_max = denoised_action.shape[1] * input_dt
        prediction_times = torch.arange(input_dt, t_max + input_dt, output_dt)
        prediction_times = prediction_times.unsqueeze(0)
        # NOTE: index the first t_max/output_dt to counteract possible rounding errors
        traj_steps = int(t_max / output_dt)
        prediction_times = prediction_times[:, :traj_steps].to(denoised_action.device)
        extra_inputs["times"] = prediction_times

        # Decode the motion primitive for the upsampled time steps
        upsampled_action = self.model.inner_model.prodmp_handler.decode(self.model.inner_model.latest_prodmp_parameters, **extra_inputs)

        return upsampled_action
