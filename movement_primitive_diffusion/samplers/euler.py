import torch

from movement_primitive_diffusion.samplers.base_sampler import BaseSampler
from typing import Dict, Optional, Callable


class EulerSampler(BaseSampler):
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
            Denoised samples
        """

        extra_inputs = {} if extra_inputs is None else extra_inputs
        # get s_in of shape (batch_size, 1, 1) so that sigmas[i] (a float) is broadcast to the correct shape
        s_in_shape = [1] * action.dim()
        s_in_shape[0] = action.shape[0]
        s_in = action.new_ones(s_in_shape)

        for i in range(len(sigmas) - 1):
            gamma = min(self.s_churn / (len(sigmas) - 1), 2**0.5 - 1) if self.s_tmin <= sigmas[i] <= self.s_tmax else 0.0
            eps = torch.randn_like(action) * self.s_noise  # sample current noise depnding on S_noise
            sigma_hat = sigmas[i] * (gamma + 1)  # add noise to sigma

            if gamma > 0:  # if gamma > 0, use additional noise level for computation
                action = action + eps * (sigma_hat**2 - sigmas[i] ** 2) ** 0.5
            denoised = model(state, action, sigma_hat * s_in, extra_inputs)  # compute denoised action
            assert sigma_hat.ndim == 0, "sigma_hat should be a scalar"
            d = (action - denoised) / sigma_hat  # compute derivative

            if callback is not None:
                callback({"x": action, "i": i, "sigma": sigmas[i], "sigma_hat": sigma_hat, "denoised": denoised})

            dt = sigmas[i + 1] - sigma_hat  # compute timestep

            # Euler method
            action = action + d * dt  # take Euler step

        return action
