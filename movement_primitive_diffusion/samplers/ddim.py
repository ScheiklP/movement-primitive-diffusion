import torch

from movement_primitive_diffusion.samplers.base_sampler import BaseSampler
from typing import Dict, Optional, Callable

from movement_primitive_diffusion.utils.visualization import plot_diffusion_steps


class DDIMSampler(BaseSampler):
    """DPM-Solver 1 (equivalent to DDIM sampler)"""

    def __init__(self, device="cpu", dtype=torch.float32):
        super().__init__(device, dtype)

    @torch.no_grad()
    def sample(
        self,
        model: torch.nn.Module,
        state: torch.Tensor,
        action: torch.Tensor,
        sigmas: torch.Tensor,
        extra_inputs: Optional[Dict] = None,
        callback: Optional[Callable] = None,
    ):
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
        extra_inputs = {} if extra_inputs is None else extra_inputs
        # get s_in of shape (batch_size, 1, 1) so that sigmas[i] (a float) is broadcast to the correct shape
        s_in_shape = [1] * action.dim()
        s_in_shape[0] = action.shape[0]
        s_in = action.new_ones(s_in_shape)
        sigma_fn = lambda t: t.neg().exp()
        t_fn = lambda sigma: sigma.log().neg()

        for i in range(len(sigmas) - 1):
            # predict the next action
            denoised = model(state, action, sigmas[i] * s_in, extra_inputs)
            if callback is not None:
                callback({"action": action, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": denoised})
            t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
            h = t_next - t
            # NOTE: Same as sigma_next/sigma * (action - denoised) + denoised. -> for sigma_next == 0 -> action = denoised
            action = (sigma_fn(t_next) / sigma_fn(t)) * action - (-h).expm1() * denoised

        return action

    @torch.no_grad()
    def sample_and_plot(
        self,
        model: torch.nn.Module,
        state: torch.Tensor,
        action: torch.Tensor,
        sigmas: torch.Tensor,
        extra_inputs: Optional[Dict] = None,
        callback: Optional[Callable] = None,
    ):
        extra_inputs = {} if extra_inputs is None else extra_inputs
        s_in = action.new_ones([action.shape[0]])
        sigma_fn = lambda t: t.neg().exp()
        t_fn = lambda sigma: sigma.log().neg()

        action_buffer = action.clone()
        denoised_buffer = None

        for i in range(len(sigmas) - 1):
            # predict the next action
            denoised = model(state, action, sigmas[i] * s_in, extra_inputs)

            if denoised_buffer is None:
                denoised_buffer = denoised.clone()
            else:
                denoised_buffer = torch.cat([denoised_buffer, denoised], dim=0)

            if callback is not None:
                callback({"action": action, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": denoised})
            t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
            h = t_next - t
            action = (sigma_fn(t_next) / sigma_fn(t)) * action - (-h).expm1() * denoised

            action_buffer = torch.cat([action_buffer, action], dim=0)

        assert denoised_buffer is not None
        plot_diffusion_steps(denoised_buffer, action_buffer)

        return action
