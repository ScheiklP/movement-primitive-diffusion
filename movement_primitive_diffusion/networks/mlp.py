from typing import Any, Callable, Dict, List, Optional, Union
import warnings

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear


# Adapted from https://github.com/google-research/ibc/blob/master/networks/layers/resnet.py#L132
class ResidualMLPBlock(torch.nn.Module):
    def __init__(
        self,
        num_neurons: int = 100,
        hidden_nonlinearity: str = "ReLU",
        dropout_rate: float = 0.0,
        spectral_norm: bool = False,
        norm: Optional[str] = None,
    ):
        super().__init__()

        sequence = list()

        if spectral_norm:
            l1 = torch.nn.utils.spectral_norm(torch.nn.Linear(num_neurons, num_neurons))
            l2 = torch.nn.utils.spectral_norm(torch.nn.Linear(num_neurons, num_neurons))
        else:
            l1 = torch.nn.Linear(num_neurons, num_neurons)
            l2 = torch.nn.Linear(num_neurons, num_neurons)

        dropout = torch.nn.Dropout(dropout_rate)

        # get nonlinearity
        if isinstance(hidden_nonlinearity, str):
            hidden_nonlinearity: torch.nn.Module = getattr(torch.nn, hidden_nonlinearity)

        # get norm
        if norm is not None:
            if isinstance(norm, str):
                norm: torch.nn.Module = getattr(torch.nn, norm)

        # build sequence of layers [(norm), nonlinearity, dropout, linear, (norm), nonlinearity, dropout, linear]
        if norm is not None:
            sequence.append(norm())
        sequence.extend([hidden_nonlinearity(), dropout, l1])
        if norm is not None:
            sequence.append(norm())
        sequence.extend([hidden_nonlinearity(), dropout, l2])

        self.model = torch.nn.Sequential(*sequence)

    def forward(self, x):
        x_input = x
        x = self.model(x)
        return x + x_input


class SimpleMLP(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        num_layers: int,
        num_neurons: int,
        output_size: Optional[int],  # if None, last layer has nonlinearity applied.
        hidden_nonlinearity: Union[str, torch.nn.Module] = "LeakyReLU",  # Module, not Functional.
    ):
        super().__init__()
        # get nonlinearity
        if isinstance(hidden_nonlinearity, str):
            hidden_nonlinearity: torch.nn.Module = getattr(torch.nn, hidden_nonlinearity)
        hidden_sizes = [num_neurons] * num_layers
        hidden_layers = [torch.nn.Linear(n_in, n_out) for n_in, n_out in zip([input_size] + hidden_sizes[:-1], hidden_sizes)]
        sequence = list()
        for layer in hidden_layers:
            sequence.extend([layer, hidden_nonlinearity()])
        if output_size is not None:
            last_size = hidden_sizes[-1] if hidden_sizes else input_size
            sequence.append(torch.nn.Linear(last_size, output_size))
        self.model = torch.nn.Sequential(*sequence)
        self._output_size = hidden_sizes[-1] if output_size is None else output_size

    def forward(self, input: Tensor) -> Tensor:
        """Compute the model on the input, assuming input shape [B,input_size]."""
        return self.model(input)


class IndependentTimeStepsMLP(SimpleMLP):
    def __init__(
        self,
        feature_size: int,
        num_layers: int,
        num_neurons: int,
        output_size: Optional[int] = None,  # if None, last layer has nonlinearity applied.
        hidden_nonlinearity: Union[str, torch.nn.Module] = "LeakyReLU",  # Module, not Functional.
        flatten_time: bool = True,
    ):
        if not isinstance(feature_size, int):
            assert len(feature_size) == 1
            feature_size = feature_size[0]

        super().__init__(
            input_size=feature_size,
            num_layers=num_layers,
            num_neurons=num_neurons,
            output_size=output_size,
            hidden_nonlinearity=hidden_nonlinearity,
        )

        self.flatten_time = flatten_time

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        assert x.dim() == 3

        # Move time dimension into batch dimension to process time steps individually (B, T, F) -> (B * T, F)
        original_shape = x.shape
        x = x.reshape(original_shape[0] * original_shape[1], -1)

        # Forward pass (B * T, F) -> (B * T, out_channels)
        output = super().forward(x)

        if self.flatten_time:
            # Reshape output to (B, T*out_channels)
            output = output.view(original_shape[0], original_shape[1] * output.size(1))
        else:
            # Reshape output to (B, T, out_channels)
            output = output.view(original_shape[0], original_shape[1], output.size(1))

        return output


class DependentTimeStepsMLP(SimpleMLP):
    def __init__(
        self,
        feature_size: int,
        num_layers: int,
        num_neurons: int,
        output_size: Optional[int] = None,  # if None, last layer has nonlinearity applied.
        time_steps: Optional[int] = None,
        hidden_nonlinearity: Union[str, torch.nn.Module] = "LeakyReLU",  # Module, not Functional.
    ):
        if not isinstance(feature_size, int):
            assert len(feature_size) == 1
            feature_size = feature_size[0]

        if time_steps is not None:
            input_size = feature_size * time_steps
        else:
            input_size = feature_size

        self.effective_input_size = input_size

        super().__init__(
            input_size=input_size,
            num_layers=num_layers,
            num_neurons=num_neurons,
            output_size=output_size,
            hidden_nonlinearity=hidden_nonlinearity,
        )

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        assert x.dim() == 3

        # Move time dimension into feature dimension to process time steps together (B, T, F) -> (B, F*T)
        x = x.view(x.size(0), -1)

        assert x.size(1) == self.effective_input_size

        # Forward pass (B, F*T) -> (B, out_channels)
        output = super().forward(x)

        return output


class MLP(torch.nn.Module):
    r"""A Multi-Layer Perception (MLP) model.
    There exists two ways to instantiate an :class:`MLP`:

    1. By specifying explicit channel sizes, *e.g.*,

       .. code-block:: python

          mlp = MLP([16, 32, 64, 128])

       creates a three-layer MLP with **differently** sized hidden layers.

    1. By specifying fixed hidden channel sizes over a number of layers,
       *e.g.*,

       .. code-block:: python

          mlp = MLP(in_channels=16, hidden_channels=32,
                    out_channels=128, num_layers=3)

       creates a three-layer MLP with **equally** sized hidden layers.

    Args:
        channel_list (List[int] or int, optional): List of input, intermediate
            and output channels such that :obj:`len(channel_list) - 1` denotes
            the number of layers of the MLP (default: :obj:`None`)
        in_channels (int, optional): Size of each input sample.
            Will override :attr:`channel_list`. (default: :obj:`None`)
        hidden_channels (int, optional): Size of each hidden sample.
            Will override :attr:`channel_list`. (default: :obj:`None`)
        out_channels (int, optional): Size of each output sample.
            Will override :attr:`channel_list`. (default: :obj:`None`)
        num_layers (int, optional): The number of layers.
            Will override :attr:`channel_list`. (default: :obj:`None`)
        dropout (float or List[float], optional): Dropout probability of each
            hidden embedding. If a list is provided, sets the dropout value per
            layer. (default: :obj:`0.`)
        act (str or Callable, optional): The non-linear activation function to
            use. (default: :obj:`"relu"`)
        act_first (bool, optional): If set to :obj:`True`, activation is
            applied before normalization. (default: :obj:`False`)
        act_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective activation function defined by :obj:`act`.
            (default: :obj:`None`)
        norm (str or Callable, optional): The normalization function to
            use. (default: :obj:`"batch_norm"`)
        norm_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective normalization function defined by :obj:`norm`.
            (default: :obj:`None`)
        plain_last (bool, optional): If set to :obj:`False`, will apply
            non-linearity, batch normalization and dropout to the last layer as
            well. (default: :obj:`True`)
        bias (bool or List[bool], optional): If set to :obj:`False`, the module
            will not learn additive biases. If a list is provided, sets the
            bias per layer. (default: :obj:`True`)
        **kwargs (optional): Additional deprecated arguments of the MLP layer.
    """

    def __init__(
        self,
        channel_list: Optional[Union[List[int], int]] = None,
        *,
        in_channels: Optional[int] = None,
        hidden_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        num_layers: Optional[int] = None,
        dropout: Union[float, List[float]] = 0.0,
        act: Union[str, Callable, None] = "ReLU",
        act_first: bool = False,
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = None,
        norm_kwargs: Optional[Dict[str, Any]] = None,
        plain_last: bool = True,
        bias: Union[bool, List[bool]] = True,
        **kwargs,
    ):
        super().__init__()

        # Backward compatibility:
        act_first = act_first or kwargs.get("relu_first", False)
        batch_norm = kwargs.get("batch_norm", None)
        if batch_norm is not None and isinstance(batch_norm, bool):
            warnings.warn("Argument `batch_norm` is deprecated, " "please use `norm` to specify normalization layer.")
            norm = "batch_norm" if batch_norm else None
            batch_norm_kwargs = kwargs.get("batch_norm_kwargs", None)
            norm_kwargs = batch_norm_kwargs or {}

        if isinstance(channel_list, int):
            in_channels = channel_list

        if in_channels is not None:
            if num_layers is None:
                raise ValueError("Argument `num_layers` must be given")
            if num_layers > 1 and hidden_channels is None:
                raise ValueError(f"Argument `hidden_channels` must be given " f"for `num_layers={num_layers}`")
            if out_channels is None:
                raise ValueError("Argument `out_channels` must be given")

            channel_list = [hidden_channels] * (num_layers - 1)
            channel_list = [in_channels] + channel_list + [out_channels]

        assert isinstance(channel_list, (tuple, list))
        assert len(channel_list) >= 2
        self.channel_list = channel_list

        self.act = getattr(torch.nn, act)(**(act_kwargs or {}))
        self.act_first = act_first
        self.plain_last = plain_last

        if isinstance(dropout, float):
            dropout = [dropout] * (len(channel_list) - 1)
            if plain_last:
                dropout[-1] = 0.0
        if len(dropout) != len(channel_list) - 1:
            raise ValueError(f"Number of dropout values provided ({len(dropout)} does not " f"match the number of layers specified " f"({len(channel_list)-1})")
        self.dropout = dropout

        if isinstance(bias, bool):
            bias = [bias] * (len(channel_list) - 1)
        if len(bias) != len(channel_list) - 1:
            raise ValueError(f"Number of bias values provided ({len(bias)}) does not match " f"the number of layers specified ({len(channel_list)-1})")

        self.lins = torch.nn.ModuleList()
        iterator = zip(channel_list[:-1], channel_list[1:], bias)
        for in_channels, out_channels, _bias in iterator:
            self.lins.append(Linear(in_channels, out_channels, bias=_bias))

        self.norms = torch.nn.ModuleList()
        iterator = channel_list[1:-1] if plain_last else channel_list[1:]
        for hidden_channels in iterator:
            if norm is not None:
                norm_layer = getattr(torch.nn, norm)(
                    hidden_channels,
                    **(norm_kwargs or {}),
                )
            else:
                norm_layer = torch.nn.Identity()
            self.norms.append(norm_layer)

        self.reset_parameters()

    @property
    def in_channels(self) -> int:
        r"""Size of each input sample."""
        return self.channel_list[0]

    @property
    def out_channels(self) -> int:
        r"""Size of each output sample."""
        return self.channel_list[-1]

    @property
    def num_layers(self) -> int:
        r"""The number of layers."""
        return len(self.channel_list) - 1

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        for lin in self.lins:
            lin.reset_parameters()
        for norm in self.norms:
            if hasattr(norm, "reset_parameters"):
                norm.reset_parameters()

    def forward(
        self,
        x: Tensor,
        return_emb: Optional[Tensor] = None,
    ) -> Tensor:
        r"""
        Args:
            x (torch.Tensor): The source tensor.
            return_emb (bool, optional): If set to :obj:`True`, will
                additionally return the embeddings before execution of the
                final output layer. (default: :obj:`False`)
        """
        emb: Optional[Tensor] = None

        for i, (lin, norm) in enumerate(zip(self.lins, self.norms)):
            x = lin(x)
            if self.act is not None and self.act_first:
                x = self.act(x)
            x = norm(x)
            if self.act is not None and not self.act_first:
                x = self.act(x)
            x = F.dropout(x, p=self.dropout[i], training=self.training)
            if isinstance(return_emb, bool) and return_emb is True:
                emb = x

        if self.plain_last:
            x = self.lins[-1](x)
            x = F.dropout(x, p=self.dropout[-1], training=self.training)

        return (x, emb) if isinstance(return_emb, bool) else x

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({str(self.channel_list)[1:-1]})"
