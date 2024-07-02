from typing import Tuple, Union, List

import torch
from torch import nn


class LSTMGMMInnerModel(torch.nn.Module):

    def __init__(
            self,
            action_size: int,
            state_size: int,
            t_pred: int,
            num_layers: int,
            num_modes: int,
            mlp_dims: List[int],
            rnn_hidden_dim: int,
    ):
        super().__init__()
        self.action_size = action_size
        self.state_size = state_size
        self.t_pred = t_pred
        self.hidden_size = rnn_hidden_dim

        self.num_layers = num_layers
        self.num_modes = num_modes

        self.mean_size = self.action_size * self.num_modes
        self.scales_size = self.action_size * self.num_modes
        self.logits_size = self.num_modes
        self.mlp_output_size = self.mean_size + self.scales_size + self.logits_size

        self.lstm = nn.LSTM(
            input_size=self.state_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
        )

        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, mlp_dims[0]),
            nn.ReLU(),
        )

        for i in range(1, len(mlp_dims)):
            self.mlp.add_module(
                f"mlp_{i}",
                nn.Linear(mlp_dims[i - 1], mlp_dims[i]),
            )
            self.mlp.add_module(
                f"relu_{i}",
                nn.ReLU(),
            )

        self.mlp.add_module(
            "mlp_out",
            nn.Linear(mlp_dims[-1], self.mlp_output_size),
        )


    def forward(self, state: torch.Tensor) -> torch.distributions.MixtureSameFamily:
        """
        Forward pass of the model
        Args:
            state: state tensor [batch_size, obs_dim]

        Returns:
            Predicted action tensor [batch_size, action_dim]
        """
        batch_size = state.shape[0]

        lstm_inputs = state.unsqueeze(1).repeat(1, self.t_pred, 1)
        lstm_out, _ = self.lstm(lstm_inputs)

        mlp_out = self.mlp(lstm_out)

        means = mlp_out[:, :, :self.mean_size].view(batch_size, self.t_pred, self.num_modes, self.action_size)
        scales = mlp_out[:, :, self.mean_size:self.mean_size + self.scales_size].view(batch_size, self.t_pred,
                                                                                       self.num_modes, self.action_size)
        logits = mlp_out[:, :, self.mean_size + self.scales_size:].view(batch_size, self.t_pred, self.num_modes)

        scales = torch.nn.functional.softplus(scales)

        component_distribution = torch.distributions.Normal(loc=means, scale=scales)
        component_distribution = torch.distributions.Independent(component_distribution, 1)
        mixture_distribution = torch.distributions.Categorical(logits=logits)

        dists = torch.distributions.MixtureSameFamily(mixture_distribution, component_distribution)

        return dists
