import torch

from typing import Union, Optional, Tuple

from movement_primitive_diffusion.networks.mlp import SimpleMLP


class Decoder(torch.nn.Module):
    """Decoder to decode latent variable to target value distribution.

    Args:
        latent_variable_size (int): Latent size of aggregated latent observation vector as input to decoder
        additional_input_size (int): Additional input dimension
        output_size (int): Size of output mean vector (dimension of the output space)
        hidden_sizes_mean_network (Union[int, List[int], None]): Hidden sizes of mean network
        hidden_sizes_covariance_network (Union[int, List[int], None]): Hidden sizes of covariance network
        hidden_nonlinearity_mean_network (torch.nn.Module, optional): Hidden nonlinearity of mean network. Defaults to torch.nn.LeakyReLU.
        hidden_nonlinearity_covariance_network (torch.nn.Module, optional): Hidden nonlinearity of covariance network. Defaults to torch.nn.LeakyReLU.
        std_only (bool, optional): Whether to only predict the standard deviation instead of the cholesky decomposition. Defaults to False.
        log_lower_bound (float, optional): Lower bound for log-space. Defaults to 1e-8.
        softplus_lower_bound (float, optional): Lower bound for softplus-space. Defaults to 1e-2.
    """

    def __init__(
        self,
        latent_variable_size: int,
        additional_input_size: int,
        output_size: int,
        num_layers_mean_network: int,
        num_layers_covariance_network: int,
        num_neurons_mean_network: int,
        num_neurons_covariance_network: int,
        hidden_nonlinearity_mean_network: Union[str, torch.nn.Module] = "LeakyReLU",
        hidden_nonlinearity_covariance_network: Union[str, torch.nn.Module] = "LeakyReLU",
        std_only: bool = False,
        log_lower_bound: float = 1e-8,
        softplus_lower_bound: float = 1e-2,
        **kwargs,
    ):
        super().__init__()

        # get nonlinearities
        if isinstance(hidden_nonlinearity_mean_network, str):
            hidden_nonlinearity_mean_network: torch.nn.Module = getattr(torch.nn, hidden_nonlinearity_mean_network)
        if isinstance(hidden_nonlinearity_covariance_network, str):
            hidden_nonlinearity_covariance_network: torch.nn.Module = getattr(torch.nn, hidden_nonlinearity_covariance_network)

        # Latent size of aggregated latent observation vector as input to decoder
        self.latent_variable_size = latent_variable_size

        # Additional input dimension
        self.additional_input_size = additional_input_size

        # Size of output mean vector (dimension of the output space)
        self.output_size = output_size

        # Whether to only predict the standard deviation instead of the cholesky decomposition
        self.std_only = std_only

        # Lower bound for log-space
        self.log_lower_bound = log_lower_bound

        # Lower bound for softplus-space
        self.softplus_lower_bound = softplus_lower_bound
        self.softplus = torch.nn.Softplus()

        # Neural networks
        self.mean_network = SimpleMLP(
            input_size=latent_variable_size + additional_input_size,
            output_size=output_size,
            num_layers=num_layers_mean_network,
            num_neurons=num_neurons_mean_network,
            hidden_nonlinearity=hidden_nonlinearity_mean_network,
        )

        # Compute the output dimension of covariance network
        if self.std_only:
            # Only has diagonal elements
            output_size_covariance_network = output_size
        else:
            # Diagonal + Non-diagonal elements, form up Cholesky Decomposition
            output_size_covariance_network = output_size + (output_size * (output_size - 1)) // 2

        self.covariance_network = SimpleMLP(
            input_size=latent_variable_size + additional_input_size,
            output_size=output_size_covariance_network,
            num_layers=num_layers_covariance_network,
            num_neurons=num_neurons_covariance_network,
            hidden_nonlinearity=hidden_nonlinearity_covariance_network,
        )

    def forward(
        self,
        latent_mean: torch.Tensor,
        latent_variance: torch.Tensor,
        additional_inputs: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Decode and compute target value's distribution

        Args:
            latent_mean (torch.Tensor): mean of latent variable
                with shape [batch_size, latent_variable_size]
            latent_variance (torch.Tensor): variance of latent variable
                with shape [batch_size, latent_variable_size]
            additional_inputs (Optional[torch.Tensor]): additional inputs, can be None.
                Shape: [batch_size, additional_input_size]

        Returns:
            mean_values (torch.Tensor): mean of target value
                with shape [batch_size, output_size]
            diagonal_covariance_values (torch.Tensor): diagonal elements of Cholesky Decomposition of covariance of target value
                with shape [batch_size, output_size]
            off_diagonal_covariance_values (Optional[torch.Tensor]): None, or off-diagonal elements of Cholesky Decomposition of covariance of target value
                with shape [batch_size, (output_size * (output_size - 1) // 2)]

        """

        # Dimension check
        assert latent_mean.ndim == latent_variance.ndim == 2

        # Process additional_inputs
        if additional_inputs is not None:
            assert additional_inputs.ndim == 2

        # Parametrize variance
        assert latent_variance.min() >= 0 and isinstance(latent_variance, torch.Tensor)
        log_latent_variance = torch.log(latent_variance + self.log_lower_bound)

        # Prepare input to decoder networks
        mean_network_input = latent_mean
        covariance_network_input = log_latent_variance

        if additional_inputs is not None:
            mean_network_input = torch.cat((additional_inputs, mean_network_input), dim=-1)
            covariance_network_input = torch.cat((additional_inputs, covariance_network_input), dim=-1)

        # Decode
        mean_values = self.mean_network(mean_network_input)
        covariance_values = self.covariance_network(covariance_network_input)

        # Process covariance net prediction
        # Decompose diagonal and off-diagonal elements
        diagonal_covariance_values = covariance_values[..., : self.output_size]
        off_diagonal_covariance_values = None if self.std_only else covariance_values[..., self.output_size :]

        # De-parametrize Log-Cholesky for diagonal elements
        diagonal_covariance_values = self.softplus(diagonal_covariance_values)
        diagonal_covariance_values = diagonal_covariance_values + self.softplus_lower_bound

        # Return
        return mean_values, diagonal_covariance_values, off_diagonal_covariance_values
