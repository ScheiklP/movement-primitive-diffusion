import torch
from typing import Optional, Union


# see see https://github.com/ALRhub/ProDMP_RAL/blob/main/nmp/aggregator.py
class MeanAggregator:
    """A mean aggregator

    This aggregator aggregates the latent observation by taking the mean of
    the latent observation of all samples in a mini batch. The aggregator
    maintains a history of the aggregated latent observation sets, which is
    updated after each aggregation.

    Args:
        latent_variable_size (int): size of the latent variable
        prior_mean (float): mean of prior distribution of latent variable

    """

    def __init__(
        self,
        latent_variable_size: int,
        prior_mean: float,
        device: Union[str, torch.device],
    ) -> None:
        # Aggregator dimension
        self.latent_variable_size = latent_variable_size

        # Scalar prior
        self.prior_mean_init = prior_mean

        # Batch size, i.e. number of trajectories
        self.batch_size: Union[int, None] = None

        # Number of aggregated subsets, each subset may contain more than 1 obs
        self.number_of_aggregations = 0

        # Aggregation history of latent variables
        self.mean_latent_variable_state: Union[torch.Tensor, None] = None

        self.device = device

    def reset(self, batch_size: int) -> None:
        """Reset the aggregator

         Note:
             Reset aggregation history of latent observation
             i.e. mean_latent_variable_state

             Note its shape[1] = number_of_aggregations + 1, which tells how many context
             "sets" have been aggregated by the aggregator, e.g. index 0
             denotes the prior mean of latent observation, index -1 denotes the
             current mean of latent observation. Note in each aggregation, the
             latent observation to be aggregated may have different number of
             samples.

        Args:
            batch_size (int): batch size
        """

        # Reset batch_size, i.e. equals to batch size
        self.batch_size = batch_size

        # Reset number of counters
        self.number_of_aggregations = 0
        self.number_of_aggregated_observations = 0

        # Shape of mean_latent_variable_state: [batch_size, number_of_aggregations + 1, latent_variable_size]

        # Get prior tensors from scalar
        prior_mean = self.generate_prior(self.prior_mean_init)

        # Add one axis (record number of aggregation)
        self.mean_latent_variable_state = prior_mean[:, None, :]

    def generate_prior(self, mean: float) -> torch.Tensor:
        """Given scalar value of mean, generate prior tensor

        Args:
            mean (float): scalar value of mean

        Returns:
            prior_mean (torch.Tensor): tensors of prior's mean for mean latent observation with shape [batch_size, latent_variable_size]
        """

        if self.batch_size is None:
            raise ValueError(
                "Batch size is not set. Please reset the aggregator first."
            )

        prior_mean = torch.full(
            size=(self.batch_size, self.latent_variable_size), fill_value=mean
        )

        prior_mean = prior_mean.to(self.device)

        return prior_mean

    def aggregate(self, latent_observation: torch.Tensor) -> None:
        """Aggregate info of latent observation

        Args:
            latent_observation (torch.Tensor): latent observations of samples of the mini batch with shape [batch_size, number_of_observations, latent_variable_size]

        """

        if self.mean_latent_variable_state is None:
            raise ValueError(
                "Aggregator has not been reset. Please reset the aggregator first."
            )

        # Check input shapes
        assert latent_observation.ndim == 3
        assert latent_observation.shape[0] == self.batch_size
        assert latent_observation.shape[2] == self.latent_variable_size

        # Number of observations
        number_of_observations = latent_observation.shape[1]

        # Get latest latent obs
        # Shape of mean_latent_variable_state: [batch_size, number_of_aggregations + 1, latent_variable_size]
        mean_latent_variable = self.mean_latent_variable_state[:, -1, :]

        mean_latent_variable = (
            mean_latent_variable * self.number_of_aggregated_observations
            + torch.sum(latent_observation, dim=1)
        ) / (self.number_of_aggregated_observations + number_of_observations)

        # Append
        self.mean_latent_variable_state = torch.cat(
            (self.mean_latent_variable_state, mean_latent_variable[:, None, :]), dim=1
        )

        # Update counters
        self.number_of_aggregations += 1
        self.number_of_aggregated_observations += number_of_observations

    def get_aggregator_state(self, index: Optional[int] = None) -> torch.Tensor:
        """Get the aggregator's state

        Return the all latent observation states, or the one at given index.
        E.g. index -1 denotes the last latent obs state; index 0 the prior

        Args:
            index (int): index of the latent observation state to be returned

        Returns:
            mean_latent_variable_state (torch.Tensor): mean of the latent observation state with shape [batch_size, number_of_aggregations + 1, latent_variable_size]
        """

        if self.mean_latent_variable_state is None:
            raise ValueError(
                "Aggregator has not been reset. Please reset the aggregator first."
            )

        if index is None:
            # Full case
            return self.mean_latent_variable_state
        elif index == -1 or index + 1 == self.mean_latent_variable_state.shape[1]:
            # Index case -1
            return self.mean_latent_variable_state[:, index:, :]
        else:
            # Other index cases
            return self.mean_latent_variable_state[:, index : index + 1, :]

    def to(self, device: Union[str, torch.device]) -> None:
        """Move aggregator to device.

        Args:
            device (str): Device to move encoder to.
        """

        if self.mean_latent_variable_state is None:
            raise ValueError(
                "Aggregator has not been reset yet. Please reset the aggregator first."
            )

        self.mean_latent_variable_state = self.mean_latent_variable_state.to(device)
