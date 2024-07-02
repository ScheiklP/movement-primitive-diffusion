import torch
import numpy as np
import math
from abc import ABC, abstractmethod
from typing import List


class Distribution(ABC):
    def __init__(self, device="cpu", dtype=torch.float32):
        super().__init__()
        self.device = device
        self.dtype = dtype

    @abstractmethod
    def sample(self, shape: List[int]) -> torch.Tensor:
        raise NotImplementedError


class RandLogNormal(Distribution):
    def __init__(self, loc=0.0, scale=1.0, device="cpu", dtype=torch.float32):
        super().__init__(device=device, dtype=dtype)
        self.loc = loc
        self.scale = scale

    def sample(self, shape: List[int]) -> torch.Tensor:
        """Draws samples from a lognormal distribution."""
        return (torch.randn(shape, device=self.device, dtype=self.dtype) * self.scale + self.loc).exp()


class RandLogLogistic(Distribution):
    def __init__(self, loc=0.0, scale=1.0, min_value=0.0, max_value=float("inf"), device="cpu", dtype=torch.float32):
        super().__init__(device=device, dtype=dtype)
        self.loc = loc
        self.scale = scale
        self.min_value = min_value
        self.max_value = max_value

    def sample(self, shape: List[int]) -> torch.Tensor:
        """Draws samples from an optionally truncated log-logistic distribution."""
        min_value = torch.as_tensor(self.min_value, device=self.device, dtype=torch.float64)
        max_value = torch.as_tensor(self.max_value, device=self.device, dtype=torch.float64)
        min_cdf = min_value.log().sub(self.loc).div(self.scale).sigmoid()
        max_cdf = max_value.log().sub(self.loc).div(self.scale).sigmoid()
        u = torch.rand(shape, device=self.device, dtype=torch.float64) * (max_cdf - min_cdf) + min_cdf
        return u.logit().mul(self.scale).add(self.loc).exp().to(self.dtype)


class RandLogUniform(Distribution):
    def __init__(self, min_value, max_value, device="cpu", dtype=torch.float32):
        super().__init__(device=device, dtype=dtype)
        self.min_value = min_value
        self.max_value = max_value

    def sample(self, shape: List[int]) -> torch.Tensor:
        """Draws samples from an log-uniform distribution."""
        min_value = math.log(self.min_value)
        max_value = math.log(self.max_value)
        return (torch.rand(shape, device=self.device, dtype=self.dtype) * (max_value - min_value) + min_value).exp()


class RandNormal(Distribution):
    def __init__(self, loc=0.0, scale=1.0, device="cpu", dtype=torch.float32):
        super().__init__(device=device, dtype=dtype)
        self.loc = loc
        self.scale = scale

    def sample(self, shape: List[int]) -> torch.Tensor:
        """Draws samples from a normal distribution."""
        return torch.randn(shape, device=self.device, dtype=self.dtype) * self.scale + self.loc


class RandUniform(Distribution):
    def __init__(self, min_value, max_value, device="cpu", dtype=torch.float32):
        super().__init__(device=device, dtype=dtype)
        self.min_value = min_value
        self.max_value = max_value

    def sample(self, shape: List[int]) -> torch.Tensor:
        """Draws samples from an uniform distribution."""
        return torch.rand(shape, device=self.device, dtype=self.dtype) * (self.max_value - self.min_value) + self.min_value


class RandDiscrete(Distribution):
    def __init__(self, values, device="cpu", dtype=torch.float32):
        super().__init__(device=device, dtype=dtype)
        self.values = values

    def sample(self, shape: List[int]) -> torch.Tensor:
        """Draws samples from an discrete distribution."""
        probs = [1 / len(self.values)] * len(self.values)  # set equal probability for all values
        return torch.tensor(np.random.choice(self.values, size=shape, p=probs), device=self.device, dtype=self.dtype)


class RandDiscreteUniform(Distribution):
    def __init__(self, min_value, max_value, device="cpu", dtype=torch.float32):
        super().__init__(device=device, dtype=dtype)
        self.min_value = min_value
        self.max_value = max_value

    def sample(self, shape: List[int]) -> torch.Tensor:
        """Draws samples from an discrete uniform distribution."""
        return torch.randint(low=self.min_value, high=self.max_value, size=shape, device=self.device, dtype=self.dtype)


class RandDiscreteLogUniform(Distribution):
    def __init__(self, min_value, max_value, device="cpu", dtype=torch.float32):
        super().__init__(device=device, dtype=dtype)
        self.min_value = min_value
        self.max_value = max_value

    def sample(self, shape: List[int]) -> torch.Tensor:
        """Draws samples from an discrete log-uniform distribution."""
        return torch.randint(low=math.log(self.min_value), high=math.log(self.max_value), size=shape, device=self.device, dtype=self.dtype).exp()


class RandDiscreteLogLogistic(Distribution):
    def __init__(self, loc=0.0, scale=1.0, min_value=0.0, max_value=float("inf"), device="cpu", dtype=torch.float32):
        super().__init__(device=device, dtype=dtype)
        self.loc = loc
        self.scale = scale
        self.min_value = min_value
        self.max_value = max_value

    def sample(self, shape: List[int]) -> torch.Tensor:
        """Draws samples from an optionally truncated discrete log-logistic distribution."""
        min_value = torch.as_tensor(self.min_value, device=self.device, dtype=torch.float64)
        max_value = torch.as_tensor(self.max_value, device=self.device, dtype=torch.float64)
        min_cdf = min_value.log().sub(self.loc).div(self.scale).sigmoid()
        max_cdf = max_value.log().sub(self.loc).div(self.scale).sigmoid()
        u = torch.rand(shape, device=self.device, dtype=torch.float64) * (max_cdf - min_cdf) + min_cdf
        return u.logit().mul(self.scale).add(self.loc).exp().to(self.dtype)


class RandDiscreteLogNormal(Distribution):
    def __init__(self, loc=0.0, scale=1.0, device="cpu", dtype=torch.float32):
        super().__init__(device=device, dtype=dtype)
        self.loc = loc
        self.scale = scale

    def sample(self, shape: List[int]) -> torch.Tensor:
        """Draws samples from a discrete log-normal distribution."""
        return torch.log(torch.randn(shape, device=self.device, dtype=self.dtype) * self.scale + self.loc)


class RandDiscreteNormal(Distribution):
    def __init__(self, loc=0.0, scale=1.0, device="cpu", dtype=torch.float32):
        super().__init__(device=device, dtype=dtype)
        self.loc = loc
        self.scale = scale

    def sample(self, shape: List[int]) -> torch.Tensor:
        """Draws samples from a discrete normal distribution."""
        return torch.randn(shape, device=self.device, dtype=self.dtype) * self.scale + self.loc


class RandDiscreteTruncatedNormal(Distribution):
    def __init__(self, loc=0.0, scale=1.0, min_value=0.0, max_value=float("inf"), device="cpu", dtype=torch.float32):
        super().__init__(device=device, dtype=dtype)
        self.loc = loc
        self.scale = scale
        self.min_value = min_value
        self.max_value = max_value

    def sample(self, shape: List[int]) -> torch.Tensor:
        """Draws samples from a truncated discrete normal distribution."""
        min_value = torch.as_tensor(self.min_value, device=self.device, dtype=torch.float64)
        max_value = torch.as_tensor(self.max_value, device=self.device, dtype=torch.float64)
        min_cdf = (min_value - self.loc) / self.scale
        max_cdf = (max_value - self.loc) / self.scale
        u = torch.rand(shape, device=self.device, dtype=torch.float64) * (max_cdf - min_cdf) + min_cdf
        return u.mul(self.scale).add(self.loc).to(self.dtype)


class RandDiscreteTruncatedLogNormal(Distribution):
    def __init__(self, loc=0.0, scale=1.0, min_value=0.0, max_value=float("inf"), device="cpu", dtype=torch.float32):
        super().__init__(device=device, dtype=dtype)
        self.loc = loc
        self.scale = scale
        self.min_value = min_value
        self.max_value = max_value

    def sample(self, shape: List[int]) -> torch.Tensor:
        """Draws samples from a truncated discrete log-normal distribution."""
        min_value = torch.as_tensor(self.min_value, device=self.device, dtype=torch.float64)
        max_value = torch.as_tensor(self.max_value, device=self.device, dtype=torch.float64)
        min_cdf = (min_value.log() - self.loc) / self.scale
        max_cdf = (max_value.log() - self.loc) / self.scale
        u = torch.rand(shape, device=self.device, dtype=torch.float64) * (max_cdf - min_cdf) + min_cdf
        return u.mul(self.scale).add(self.loc).exp().to(self.dtype)


class RandVDiffusion(Distribution):
    def __init__(self, sigma_data=1.0, min_value=0.0, max_value=float("inf"), device="cpu", dtype=torch.float32):
        super().__init__(device=device, dtype=dtype)
        self.sigma_data = sigma_data
        self.min_value = min_value
        self.max_value = max_value

    def sample(self, shape: List[int]) -> torch.Tensor:
        """Draws samples from a truncated v-diffusion training timestep distribution."""
        min_cdf = math.atan(self.min_value / self.sigma_data) * 2 / math.pi
        max_cdf = math.atan(self.max_value / self.sigma_data) * 2 / math.pi
        u = torch.rand(shape, device=self.device, dtype=self.dtype) * (max_cdf - min_cdf) + min_cdf
        return torch.tan(u * math.pi / 2) * self.sigma_data


class RandSplitLogNormal(Distribution):
    def __init__(self, loc, scale_1, scale_2, device="cpu", dtype=torch.float32):
        super().__init__(device=device, dtype=dtype)
        self.loc = loc
        self.scale_1 = scale_1
        self.scale_2 = scale_2

    def sample(self, shape: List[int]) -> torch.Tensor:
        """Draws samples from a split lognormal distribution."""
        n = torch.randn(shape, device=self.device, dtype=self.dtype).abs()
        u = torch.rand(shape, device=self.device, dtype=self.dtype)
        return torch.where(u < 0.5, n.mul(self.scale_1).add(self.loc).exp(), n.mul(self.scale_2).add(self.loc).exp())
