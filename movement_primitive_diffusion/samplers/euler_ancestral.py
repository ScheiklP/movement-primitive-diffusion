import torch

from movement_primitive_diffusion.samplers.base_sampler import BaseSampler
from typing import Dict, Optional, Callable


class EulerAncestralSampler(BaseSampler):
    """Euler sampler

    Implements a variant of Algorithm 2 (Euler steps) from Karras et al. (2022).
    Stochastic sampler, which combines a first order ODE solver with explicit Langevin-like "churn"
    of adding and removing noise.
    Every update consists of these substeps:
    1. Addition of noise given the factor eps
    2. Solving the ODE dx/dt at timestep t using the score model
    3. Take Euler step from t -> t+1 to get x_{i+1}

    In contrast to the Heun sampler, this sampler does not use a correction step.
    """

    def __init__(self, eta=1, device="cpu", dtype=torch.float32):
        super().__init__(device=device, dtype=dtype)

        self.eta = eta

    def get_ancestral_step(self, sigma_from, sigma_to):
        """Calculates the noise level (sigma_down) to step down to and the amount
        of noise to add (sigma_up) when doing an ancestral sampling step."""
        if not self.eta:
            return sigma_to, 0.
        sigma_up = min(sigma_to, self.eta * (sigma_to ** 2 * (sigma_from ** 2 - sigma_to ** 2) / sigma_from ** 2) ** 0.5)
        sigma_down = (sigma_to ** 2 - sigma_up ** 2) ** 0.5
        return sigma_down, sigma_up
    @torch.no_grad()
    def sample(
        self,
        model: torch.nn.Module,
        state: torch.Tensor,
        action: torch.Tensor,
        extra_inputs: Dict,
        sigmas,
        callback: Optional[Callable] = None,
    ) -> torch.Tensor:
        """
        Sample loop to calculate the noise free action
        Args:
            original_samples: Original samples
            noise: Noise tensor
            timesteps: Timesteps tensor

        Returns:
            Denoised samples
        """

        extra_inputs = {} if extra_inputs is None else extra_inputs
        # get s_in of shape (batch_size, 1, 1) so that sigmas[i] (a float) is broadcast to the correct shape
        s_in_shape = [1] * action.dim()
        s_in_shape[0] = action.shape[0]
        s_in = action.new_ones(s_in_shape)

        for i in range(len(sigmas) - 1):
            # compute x_{t-1}
            denoised = model(state, action, sigmas[i] * s_in, extra_inputs)
            # get ancestral steps
            sigma_down, sigma_up = self.get_ancestral_step(sigmas[i], sigmas[i + 1])
            if callback is not None:
                callback({'x': action, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
            # compute dx/dt
            d = (action - denoised) / sigmas[i]

            if callback is not None:
                callback({"x": action, "i": i, "sigma": sigmas[i], "denoised": denoised})

            # compute dt based on sigma_down value
            dt = sigma_down - sigmas[i]
            # update current action
            action = action + d * dt
            if sigma_down > 0:
                action = action + torch.randn_like(action) * sigma_up

        return action