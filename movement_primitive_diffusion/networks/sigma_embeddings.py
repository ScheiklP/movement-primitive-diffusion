import math
import torch


class LinearEmbedding(torch.nn.Module):
    """Linear embedding.

    Embeds the sigma with a single linear layer with learnable weights.

    Args:
        embedding_size: Size of the embedding.
    """

    def __init__(self, embedding_size: int):
        super().__init__()
        self.embedding_size = embedding_size
        self.embed = torch.nn.Linear(1, embedding_size)

    def forward(self, sigma: torch.Tensor) -> torch.Tensor:
        return self.embed(sigma)


class RepeatEmbedding(torch.nn.Module):
    """Repeat embedding.

    Repeats the sigma embedding multiple times.

    Args:
        embedding_size: Size of the embedding.
    """

    def __init__(self, embedding_size: int):
        super().__init__()
        self.embedding_size = embedding_size

    def forward(self, sigma: torch.Tensor) -> torch.Tensor:
        return sigma.expand(-1, self.embedding_size)


class PassThroughEmbedding(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, sigma: torch.Tensor) -> torch.Tensor:
        return sigma


class GaussianFourierProjection(torch.nn.Module):
    """Gaussian random features.

    Randomly sample weights during initialization.
    These weights are fixed during optimization and are not trainable.

    Args:
        embedding_size: Size of the embedding.
        std: Standard deviation of the Gaussian distribution.
    """

    def __init__(self, embedding_size: int, std: float = 30.0):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = torch.nn.Parameter(torch.randn(embedding_size // 2) * std, requires_grad=False)

    def forward(self, sigma: torch.Tensor) -> torch.Tensor:
        sigma_proj = sigma[:, None] * self.W[None, :] * 2 * torch.pi
        return torch.cat([torch.sin(sigma_proj), torch.cos(sigma_proj)], dim=-1)


class GaussianFourierEmbedding(torch.nn.Module):
    """Gaussian random features embedding.

    Sigma is embedded using Gaussian random features, and then passed through a
    linear layer, a non-linearity, and another linear layer.

    Args:
        embedding_size: Size of the embedding.
        std: Standard deviation of the Gaussian distribution.
    """

    def __init__(self, embedding_size: int, std: float = 30.0):
        super().__init__()
        self.embedding_size = embedding_size
        self.embed = torch.nn.Sequential(
            GaussianFourierProjection(embedding_size=embedding_size, std=std),
            torch.nn.Linear(embedding_size, 2 * embedding_size),
            torch.nn.Mish(),
            torch.nn.Linear(2 * embedding_size, embedding_size),
        )

    def forward(self, sigma: torch.Tensor) -> torch.Tensor:
        return self.embed(sigma.squeeze(-1))


class FourierFeaturesEmbedding(torch.nn.Module):
    """Fourier features embedding.

    Sigma is embedded using Fourier features with learnable weights.

    Args:
        embedding_size: Size of the embedding.
        in_features: Number of input features.
        std: Standard deviation of the Gaussian distribution.

    """

    def __init__(self, embedding_size: int, std: float = 1.0):
        super().__init__()
        assert embedding_size % 2 == 0, "Embedding size must be even."
        self.register_buffer("weight", torch.randn([embedding_size // 2, 1]) * std)
        self.embedding_size = embedding_size

    def forward(self, sigma: torch.Tensor) -> torch.Tensor:
        if len(sigma.shape) == 1:
            sigma = sigma.unsqueeze(1)
        f = 2 * torch.pi * sigma @ self.weight.T
        return torch.cat([f.cos(), f.sin()], dim=-1)


class MLPEmbedding(torch.nn.Module):
    """MLP embedding.

    Sigma is embedded using a multi-layer perceptron.
    Consists of a linear layer, a non-linearity, and another linear layer.

    Args:
        embedding_size: Size of the embedding.
    """

    def __init__(self, embedding_size: int):
        super().__init__()
        self.embedding_size = embedding_size
        self.embed = torch.nn.Sequential(
            torch.nn.Linear(1, 2 * embedding_size),
            torch.nn.SiLU(),
            torch.nn.Linear(2 * embedding_size, embedding_size),
        )

    def forward(self, sigma: torch.Tensor) -> torch.Tensor:
        return self.embed(sigma)


class SinusoidalProjection(torch.nn.Module):
    """Sinusoidal projection.

    Sigma is embedded using sinusoidal projection.

    Args:
        embedding_size: Size of the embedding.
    """

    def __init__(self, embedding_size: int):
        super().__init__()
        self.embedding_size = embedding_size

    def forward(self, sigma: torch.Tensor) -> torch.Tensor:
        device = sigma.device
        half_dim = self.embedding_size // 2
        embedding = math.log(10000) / (half_dim - 1)
        embedding = torch.exp(torch.arange(half_dim, device=device) * -embedding)
        embedding = sigma * embedding[None, :]
        embedding = torch.cat((embedding.sin(), embedding.cos()), dim=-1)
        return embedding


class SinusoidalEmbedding(torch.nn.Module):
    """Sinusoidal embedding.

    Sigma is embedded using sinusoidal projection, and then passed through a
    linear layer, a non-linearity, and another linear layer.

    Args:
        embedding_size: Size of the embedding.
    """

    def __init__(self, embedding_size: int, hidden_size_factor: int = 2):
        super().__init__()
        self.embedding_size = embedding_size
        self.embed = torch.nn.Sequential(
            SinusoidalProjection(embedding_size=embedding_size),
            torch.nn.Linear(embedding_size, embedding_size * hidden_size_factor),
            torch.nn.Mish(),
            torch.nn.Linear(embedding_size * hidden_size_factor, embedding_size),
        )

    def forward(self, sigma: torch.Tensor) -> torch.Tensor:
        return self.embed(sigma)


SIGMA_EMBEDDINGS = (
    LinearEmbedding,
    RepeatEmbedding,
    PassThroughEmbedding,
    GaussianFourierEmbedding,
    FourierFeaturesEmbedding,
    MLPEmbedding,
    SinusoidalEmbedding,
    SinusoidalProjection,
)
