import torch

from movement_primitive_diffusion.samplers.base_sampler import BaseSampler
from typing import Dict, Optional, Callable


class DPMPP2MSampler(BaseSampler):
    """DPM++(2M) sampler"""

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
        extra_inputs = {} if extra_inputs is None else extra_inputs
        s_in = action.new_ones([action.shape[0]])
        sigma_fn = lambda t: t.neg().exp()
        t_fn = lambda sigma: sigma.log().neg()
        old_denoised = None

        for i in range(len(sigmas) - 1):
            denoised = model(state, action, sigmas[i] * s_in, extra_inputs)

            if callback is not None:
                callback({"action": action, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": denoised})

            t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
            h = t_next - t
            if old_denoised is None or sigmas[i + 1] == 0:
                action = (sigma_fn(t_next) / sigma_fn(t)) * action - (-h).expm1() * denoised
            else:
                h_last = t - t_fn(sigmas[i - 1])
                r = h_last / h
                denoised_d = (1 + 1 / (2 * r)) * denoised - (1 / (2 * r)) * old_denoised
                action = (sigma_fn(t_next) / sigma_fn(t)) * action - (-h).expm1() * denoised_d

            old_denoised = denoised

        return action
