import hydra
import torch

from omegaconf import DictConfig
from typing import Dict, List

from torch.nn import ModuleDict


class Encoder(torch.nn.Module):
    def __init__(
        self,
        network_configs: DictConfig,
        aggregator_config: DictConfig,
        t_obs: int,
    ) -> None:
        super().__init__()
        self.networks = ModuleDict()
        self.data_shapes: Dict[str, Dict[str, List[int]]] = {}

        for network_config in network_configs:
            feature_size: List[int] = list(network_config.feature_size)
            observation_key: str = network_config.observation_key
            config: DictConfig = network_config.network_config

            # Instantiate the network
            self.networks[observation_key] = hydra.utils.instantiate(config)

            # Do a forward pass with dummy data
            dummy_data_shape = [1, t_obs] + feature_size
            network_output = self.networks[observation_key](torch.zeros(dummy_data_shape))

            # Check whether the network returns more than one tensor
            if isinstance(network_output, tuple):
                output_data_shape = list(network_output[0].shape)
                num_output_tensors = len(network_output)
            else:
                output_data_shape = list(network_output.shape)
                num_output_tensors = 1

            # Store information about input and outputs
            self.data_shapes[observation_key] = {"input_size": dummy_data_shape[1:], "output_size": output_data_shape[1:], "num_output_tensors": num_output_tensors}

        # Instantiate the aggregator
        self.aggregator = hydra.utils.instantiate(aggregator_config)

        # Based on the stored input and output information, create dummy data for the aggregator
        aggregator_dummy_data = {}
        for observation_key in self.networks.keys():
            if (num_tensors := self.data_shapes[observation_key]["num_output_tensors"]) > 1:
                aggregator_dummy_data[observation_key] = (torch.zeros([1] + self.data_shapes[observation_key]["output_size"]),) * num_tensors
            else:
                aggregator_dummy_data[observation_key] = torch.zeros([1] + self.data_shapes[observation_key]["output_size"])

        # Do a forward pass to determine the output size of the aggregator
        aggregator_output = self.aggregator(aggregator_dummy_data)

        # Again, if the aggregator returns more than one tensor, take the first
        if isinstance(aggregator_output, tuple):
            aggregator_output = aggregator_output[0]
        if aggregator_output.ndim not in [2, 3]:
            raise ValueError(f"Aggregator output should be of shape (B, F) or (B, T, F), but got {aggregator_output.shape}.")
        self.output_size = aggregator_output.shape[-1]

    def __call__(self, observation: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Encode observations
        network_outputs = {}
        for key, network in self.networks.items():
            assert key in observation, f"Observation does not contain key {network}"
            network_outputs[key] = network(observation[key])

        # Aggregate encodings
        encoding = self.aggregator(network_outputs)

        return encoding
