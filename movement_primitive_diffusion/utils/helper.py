import torch
import numpy as np

from typing import Dict, Union, Deque


def count_parameters(model: torch.nn.Module) -> int:
    """Counts the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def tensor_to_list(value):
    """Recursively converts tensors in a dictionary to lists."""
    if isinstance(value, torch.Tensor):
        return value.tolist()
    elif isinstance(value, dict):
        return {k: tensor_to_list(v) for k, v in value.items()}
    else:
        return value


def deque_to_array(value, add_batch_dim: bool = True):
    if isinstance(value, Deque):
        if add_batch_dim:
            return np.stack(list(value))[None, ...]
        else:
            return np.stack(list(value))
    elif isinstance(value, dict):
        return {k: deque_to_array(v) for k, v in value.items() if not k == "default_factory"}
    else:
        return value


def list_of_dicts_of_arrays_to_dict_of_arrays(value):
    # Assumes that the arrays have a leading batch dimension [1, ...]
    if isinstance(value, list) or isinstance(value, tuple):
        return {k: np.concatenate([v[k] for v in value], axis=0) for k in value[0].keys()}
    elif isinstance(value, dict):
        return {k: list_of_dicts_of_tensors_to_dict_of_tensors(v) for k, v in value.items()}
    else:
        return value


def dictionary_to_device(data: Dict[str, torch.Tensor], device: Union[str, torch.device]) -> Dict[str, torch.Tensor]:
    """Recursively moves tensors in a dictionary to the GPU."""
    return {k: v.to(device) for k, v in data.items()}


def dot_to_nested_dict(input_dict, sep="."):
    """Converts a flat dict with dot-separated keys to a nested dict

    Example:
        >>> dot_to_nested_dict({"a.b.c": 1, "a.b.d": 2})
        {"a": {"b": {"c": 1, "d": 2}}}
    """
    output_dict = {}
    for key, value in input_dict.items():
        parts = key.split(sep)
        d = output_dict
        for part in parts[:-1]:
            if part not in d:
                d[part] = dict()
            d = d[part]
        d[parts[-1]] = value
    return output_dict


def nested_dict_to_dot(input_dict, current_path=""):
    """Converts a nested dict to a flat dict with dot-separated keys

    Example:
        >>> nested_dict_to_dot({"a": {"b": {"c": 1, "d": 2}}})
        {"a.b.c": 1, "a.b.d": 2}
    """
    output_dict = {}
    for key, value in input_dict.items():
        new_key = f"{current_path}.{key}" if current_path else key
        if isinstance(value, dict):
            output_dict.update(nested_dict_to_dot(value, new_key))
        else:
            output_dict[new_key] = value
    return output_dict


def format_loss(loss: float, digits_before_comma: int = 3, digits_after_comma: int = 5) -> str:
    """Formats a loss value to a string.

    Args:
        loss (float): The loss value.
        digits_before_comma (int, optional): The number of digits before the comma. Defaults to 3.
        digits_after_comma (int, optional): The number of digits after the comma. Defaults to 5.

    Returns:
        str: The formatted loss value.

    Examples:
        >>> format_loss(0.123456789)
        '000.12345'
    """
    loss_string = str(loss)

    if "e" in loss_string:
        loss_string, exponent = loss_string.split("e")
    else:
        exponent = ""

    before_comma, after_comma = loss_string.split(".")
    before_comma = before_comma.zfill(digits_before_comma)
    after_comma = after_comma[:digits_after_comma]
    return f"{before_comma}.{after_comma}{exponent}"
