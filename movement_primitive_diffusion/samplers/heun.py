import torch

from movement_primitive_diffusion.samplers.base_sampler import BaseSampler
from typing import Dict, Optional, Callable


class HeunSampler(BaseSampler):
    """
    Implements Algorithm 2 (Heun steps) from Karras et al. (2022).
    For S_churn =0 this is an ODE solver otherwise SDE
    Every update consists of these substeps:
    1. Addition of noise given the factor eps
    2. Solving the ODE dx/dt at timestep t using the score model
    3. Take Euler step from t -> t+1 to get x_{i+1}
    4. 2nd order correction step to get x_{i+1}^{(2)}

    In contrast to the Euler variant, this variant computes a 2nd order correction step.
    """

    def __init__(self, s_churn=0.0, s_tmin=0.0, s_tmax=float("inf"), s_noise=1.0, device="cpu", dtype=torch.float32):
        super().__init__(device=device, dtype=dtype)

        self.s_churn = s_churn
        self.s_tmin = s_tmin
        self.s_tmax = s_tmax
        self.s_noise = s_noise

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
            Noisy samples
        """

        s_in = action.new_ones([action.shape[0]])
        for i in range(len(sigmas) - 1):
            gamma = min(self.s_churn / (len(sigmas) - 1), 2**0.5 - 1) if self.s_tmin <= sigmas[i] <= self.s_tmax else 0.0
            eps = torch.randn_like(action) * self.s_noise
            sigma_hat = sigmas[i] * (gamma + 1)
            # if gamma > 0, use additional noise level for computation ODE-> SDE Solver
            if gamma > 0:
                action = action + eps * (sigma_hat**2 - sigmas[i] ** 2) ** 0.5
            denoised = model(state, action, sigma_hat * s_in, extra_inputs)
            d = (action - denoised) / sigma_hat
            if callback is not None:
                callback({"x": action, "i": i, "sigma": sigmas[i], "sigma_hat": sigma_hat, "denoised": denoised})
            dt = sigmas[i + 1] - sigma_hat
            # if we only are at the last step we use an Euler step for our update otherwise the heun one
            if sigmas[i + 1] == 0:
                # Euler method
                action = action + d * dt
            else:
                # Heun's method
                action_2 = action + d * dt
                denoised_2 = model(state, action_2, sigmas[i + 1] * s_in, extra_inputs)
                d_2 = (action_2 - denoised_2) / sigmas[i + 1]
                d_prime = (d + d_2) / 2
                action = action + d_prime * dt

        return action
