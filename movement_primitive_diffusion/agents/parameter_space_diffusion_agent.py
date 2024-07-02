import torch

from typing import Dict, Tuple

from movement_primitive_diffusion.agents.diffusion_agent import DiffusionAgent


class ParameterSpaceDiffusionAgent(DiffusionAgent):
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

        # Sample random noisy parameters of shape (1, ...) and scale it by sigma_max
        noised_parameters = torch.randn((batch_size, *self.process_batch.parameter_shape), device=self.device) * self.sigma_max

        denoised_parameters = self.sampler.sample(model=self.model, state=state, action=noised_parameters, sigmas=sigmas, extra_inputs=extra_inputs)

        denoised_action = self.process_batch.parameters_to_action(denoised_parameters, extra_inputs)

        # Load back the original weights
        if self.use_ema:
            self.ema_model.restore(self.model.parameters())
            self.ema_encoder.restore(self.encoder.parameters())

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

        # Process batch to get observation, parameters and extra inputs
        parameters, observation, extra_inputs = self.process_batch(batch)

        # Sampling sigma for the noise level used in the model
        # One sigma per batch element -> shape = (batch_size, 1)
        sigma = self.sigma_distribution.sample(shape=parameters.shape[0]).to(self.device)

        with torch.no_grad():
            # Encoding observation
            state = self.encoder(observation)
            eval_loss, denoised_parameters = self.model.loss(state, parameters, sigma, extra_inputs, return_denoised=True)

            denoised_action = self.process_batch.parameters_to_action(denoised_parameters, extra_inputs)
            action = extra_inputs["action"]

            # Calculate the L2 error of the first and last action of the sequence
            start_point_deviation = torch.linalg.norm(action[:, 0, :] - denoised_action[:, 0, :], dim=-1).mean()
            end_point_deviation = torch.linalg.norm(action[:, -1, :] - denoised_action[:, -1, :], dim=-1).mean()

        # Load back the original weights
        if self.use_ema:
            self.ema_model.restore(self.model.parameters())
            self.ema_encoder.restore(self.encoder.parameters())

        return eval_loss.item(), start_point_deviation.item(), end_point_deviation.item()
