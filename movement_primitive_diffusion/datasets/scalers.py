import numpy as np
import torch
from typing import Dict, Union


def get_scaler_values(data: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Compute mean, std, min, and max of data.

    TODO:
        If the arrays get too big, implement a batched version following https://github.com/openai/baselines/blob/master/baselines/common/running_mean_std.py#L5

    Args:
        data (torch.Tensor): Data array.

    Returns:
        Dict[str, torch.Tensor]: Dictionary of mean, std, min, and max.
    """

    # Reduce all but the last dimension: (batch, time, features) -> (0, 1)
    mean_axis = tuple(range(data.ndim - 1))
    data_mean = torch.mean(data, axis=mean_axis)
    data_std = torch.std(data, axis=mean_axis)
    data_min = torch.from_numpy(np.min(data.cpu().numpy(), axis=mean_axis))
    data_max = torch.from_numpy(np.max(data.cpu().numpy(), axis=mean_axis))

    return {"mean": data_mean, "std": data_std, "min": data_min, "max": data_max}


def standardize(data: Union[torch.Tensor, np.ndarray], normalizer_values: Dict[str, Union[torch.Tensor, np.ndarray]]) -> Union[torch.Tensor, np.ndarray]:
    """Standardize data using the mean and std of the normalizer_values.

    Args:
        data (Union[torch.Tensor, np.ndarray]): Data array.
        normalizer_values (Dict[str, Union[torch.Tensor, np.ndarray]]): Dictionary of mean, std, min, and max.

    Returns:
        Union[torch.Tensor, np.ndarray]: Standardized data.
    """
    return (data - normalizer_values["mean"]) / normalizer_values["std"]


def destandardize(data: Union[torch.Tensor, np.ndarray], normalizer_values: Dict[str, Union[torch.Tensor, np.ndarray]]) -> Union[torch.Tensor, np.ndarray]:
    """Destandardize data using the mean and std of the normalizer_values.

    Args:
        data (Union[torch.Tensor, np.ndarray]): Data array.
        normalizer_values (Dict[str, Union[torch.Tensor, np.ndarray]]): Dictionary of mean, std, min, and max.

    Returns:
        Union[torch.Tensor, np.ndarray]: Destandardized data.
    """
    return data * normalizer_values["std"] + normalizer_values["mean"]


def normalize(data: Union[torch.Tensor, np.ndarray], normalizer_values: Dict[str, Union[torch.Tensor, np.ndarray]], symmetric: bool) -> Union[torch.Tensor, np.ndarray]:
    """Normalize data using the min and max of the normalizer_values.

    Args:
        data (Union[torch.Tensor, np.ndarray]): Data array.
        normalizer_values (Dict[str, Union[torch.Tensor, np.ndarray]]): Dictionary of mean, std, min, and max.
        symmetric (bool): Whether to normalize to [-1, 1] (True) or [0, 1] (False).

    Returns:
        Union[torch.Tensor, np.ndarray]: Normalized data.
    """
    normalized_data = (data - normalizer_values["min"]) / (normalizer_values["max"] - normalizer_values["min"])
    if symmetric:
        normalized_data = 2.0 * normalized_data - 1.0
    return normalized_data


def denormalize(data: Union[torch.Tensor, np.ndarray], normalizer_values: Dict[str, Union[torch.Tensor, np.ndarray]], symmetric: bool, is_velocity: bool = False) -> Union[torch.Tensor, np.ndarray]:
    """Denormalize data using the min and max of the normalizer_values to the original range

    Args:
        data (Union[torch.Tensor, np.ndarray]): Data array.
        normalizer_values (Dict[str, Union[torch.Tensor, np.ndarray]]): Dictionary of mean, std, min, and max.
        symmetric (bool): Whether to denormalize from [-1, 1] (True) or [0, 1] (False).
        is_velocity (bool): Whether the data is a velocity. If True, the data is denormalized by multiplying with (max - min) only, without adding min.

    Returns:
        Union[torch.Tensor, np.ndarray]: Denormalized data.
    """

    if symmetric:
        if is_velocity:
            data = data / 2.0
        else:
            data = (data + 1.0) / 2.0

    data = data * (normalizer_values["max"] - normalizer_values["min"])

    if not is_velocity:
        data = data + normalizer_values["min"]

    return data
