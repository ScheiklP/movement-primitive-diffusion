import torch
from typing import Tuple, Union, Dict


# see https://github.com/ALRhub/ProDMP_RAL/blob/main/nmp/aggregator.py
class BayesianAggregator:
    """A Bayesian Aggregator

    This class implements a Bayesian aggregator, which aggregates latent
    observations and computes the posterior distribution of latent variables.

    Args:
        latent_variable_size (int): size of the latent variable
        prior_mean (float): mean of prior distribution of latent variable
        prior_variance (float): variance of prior distribution of latent variable
    """

    def __init__(
        self,
        latent_variable_size: int,
        prior_mean: float,
        prior_variance: float,
    ) -> None:
        # Aggregator dimension
        self.latent_variable_size = latent_variable_size

        # Scalar prior
        self.prior_mean_init = prior_mean
        self.prior_variance_init = prior_variance
        assert self.prior_variance_init >= 0  # We only consider diagonal terms, so always be positive

        # Batch size, i.e. number of trajectories
        self.batch_size: Union[int, None] = None

        # Number of aggregated subsets, each subset may contain more than 1 observation
        self.number_of_aggregations = 0

        # Aggregation history of latent variables
        self.mean_latent_variable_state: Union[torch.Tensor, None] = None
        self.variance_latent_variable_state: Union[torch.Tensor, None] = None

        # We reset the aggregator before each call to aggregate to have the correct batch size
        self.needs_reset = True

    def reset(self, batch_size: int, device: Union[str, torch.device]) -> None:
        """Reset aggregator

        Note:
              Reset aggregation history of latent variables
              i.e. mean_latent_variable_state and variance_latent_variable_state

              Note its shape[1] = number_of_aggregations + 1, which tells how many context sets
              have been aggregated by the aggregator, e.g. index 0 denotes the prior
              distribution of latent variable, index -1 denotes the current
              distribution of latent variable. Note in each aggregation, the latent
              observation may have different number of samples.

        Args:
            batch_size (int): batch size
            device (Union[str, torch.device]): device of tensors
        """
        # Set device
        self.device = device

        # Reset batch_size, i.e. equals to batch size
        self.batch_size = batch_size

        # Reset number of counters
        self.number_of_aggregations = 0

        # Shape of mean_latent_variable_state: [batch_size, number_of_aggregations + 1, latent_variable_size]
        # Shape of variance_latent_variable_state: [batch_size, number_of_aggregations + 1, latent_variable_size]

        # Get prior tensors from scalar
        prior_mean, prior_var = self.generate_prior(self.prior_mean_init, self.prior_variance_init)

        # Add one axis (record number of aggregation)
        self.mean_latent_variable_state = prior_mean[:, None, :]
        self.variance_latent_variable_state = prior_var[:, None, :]

    def generate_prior(self, mean: float, variance: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Given scalar values of mean and variance, generate prior tensor

        Args:
            mean (float): scalar value of mean
            variance (float): scalar value of variance

        Returns:
            prior_mean, prior_variance (Tuple[torch.Tensor, torch.Tensor]): tensors of prior's mean and prior's variance with shape [batch_size, latent_variable_size]
        """

        if self.batch_size is None:
            raise ValueError("Batch size is not set yet. Please reset the aggregator first.")

        prior_mean = torch.full(size=(self.batch_size, self.latent_variable_size), fill_value=mean)
        prior_variance = torch.full(size=(self.batch_size, self.latent_variable_size), fill_value=variance)

        prior_mean = prior_mean.to(self.device)
        prior_variance = prior_variance.to(self.device)

        return prior_mean, prior_variance

    def __call__(
        self,
        latent_observations: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Aggregate info of latent observation and compute new latent variable.

        If there's no latent observation, then return prior of latent variable.

        Args:
            latent_observations (Dict[str, Tuple[torch.Tensor, torch.Tensor]]): dictionary of latent observations with key as name of latent observation and value as tuple of mean and variance of latent observation
                state_mean (torch.Tensor): latent observations of samples of the mini batch with shape [batch_size, (number_of_observations,) latent_variable_size]
                state_variance (torch.Tensor): variance (uncertainty) of latent observations of samples of the mini batch with shape [batch_size, (number_of_observations,) latent_variable_size]
        """

        for state in latent_observations.values():
            state_mean, state_variance = state

            if self.needs_reset:
                batch_size = state_mean.shape[0]
                device = state_mean.device
                self.reset(batch_size=batch_size, device=device)
                self.needs_reset = False

            # (Batch, features)
            if state_mean.ndim == 2:
                # (Batch, features) -> (Batch, 1, features)
                state_mean = state_mean.unsqueeze(1)
                state_variance = state_variance.unsqueeze(1)

            assert state_mean.ndim == state_variance.ndim == 3
            assert state_mean.shape == state_variance.shape
            assert state_mean.shape[2] == self.latent_variable_size

            # Get the latest latent variable distribution
            mean_latent_variable = self.mean_latent_variable_state[:, -1, :]
            variance_latent_variable = self.variance_latent_variable_state[:, -1, :]

            variance_latent_variable = 1 / (1 / variance_latent_variable + torch.sum(1 / state_variance, dim=1))
            mean_latent_variable = mean_latent_variable + variance_latent_variable * torch.sum(
                1 / state_variance * (state_mean - mean_latent_variable[:, None, :]),
                dim=1,
            )

            # Append to latent variable state
            self.mean_latent_variable_state = torch.cat((self.mean_latent_variable_state, mean_latent_variable[:, None, :]), dim=1)
            self.variance_latent_variable_state = torch.cat(
                (self.variance_latent_variable_state, variance_latent_variable[:, None, :]),
                dim=1,
            )

            # Update counter
            self.number_of_aggregations += 1

        # Mark the end of an aggregation, so that the aggregator is reset at the next call
        self.needs_reset = True

        # Get the latest latent variable distribution, and return it with shape (B, latent_variable_size)
        mean_latent_variable = self.mean_latent_variable_state[:, -1, :].squeeze(1)
        variance_latent_variable = self.variance_latent_variable_state[:, -1, :].squeeze(1)

        return mean_latent_variable, variance_latent_variable
