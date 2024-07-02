from typing import Dict
import torch

from movement_primitive_diffusion.aggregators.base_aggregator import Aggregator


class ConcatenateAggregator(Aggregator):
    def __call__(self, encodings: Dict[str, torch.Tensor]):
        """Concatenate the values of the given data dictionary along the last dimension.

        Args:
            data (Dict[str, torch.Tensor]): The data to aggregate.

        Examples:
            >>> encodings = {"x": torch.tensor([[1, 2, 3]]), "y": torch.tensor([[4, 5, 6]])}
            >>> ConcatenateAggregator().aggregate(encodings)
            tensor([[1, 2, 3, 4, 5, 6]])
        """

        return torch.cat(list(encodings.values()), dim=-1)
