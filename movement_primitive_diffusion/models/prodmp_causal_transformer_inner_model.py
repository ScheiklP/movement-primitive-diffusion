import hydra
from omegaconf import DictConfig
import torch

from typing import Optional, Dict, Union

from movement_primitive_diffusion.models.causal_transformer_inner_model import CausalTransformer
from movement_primitive_diffusion.utils.mp_utils import ProDMPHandler


class ProDMPCausalTransformerInnerModel(CausalTransformer):
    def __init__(
        self,
        action_size: int,
        state_size: int,
        sigma_embedding_config: DictConfig,
        prodmp_handler_config: DictConfig,
        t_pred: int,
        t_obs: int,
        n_layers: int = 8,
        n_heads: int = 4,
        embedding_size: int = 256,
        dropout_probability_embedding: float = 0,
        dropout_probability_attention: float = 0.01,
        n_cond_layers: int = 0,
    ) -> None:
        super().__init__(
            action_size=action_size,
            state_size=state_size,
            sigma_embedding_config=sigma_embedding_config,
            t_pred=t_pred,
            t_obs=t_obs,
            n_layers=n_layers,
            n_heads=n_heads,
            embedding_size=embedding_size,
            dropout_probability_embedding=dropout_probability_embedding,
            dropout_probability_attention=dropout_probability_attention,
            n_cond_layers=n_cond_layers,
        )

        # Initialize ProDMPHandler
        prodmp_handler_config.num_dof = action_size
        self.prodmp_handler: ProDMPHandler = hydra.utils.instantiate(prodmp_handler_config)

        # Add a variable to hold the latest predicted ProDMP parameters
        self.latest_prodmp_parameters: Union[None, torch.Tensor] = None

        # Overwrite the decoder head
        self.head = torch.nn.Linear(embedding_size, self.prodmp_handler.encoding_size, bias=False)

    def forward(
        self,
        state: torch.Tensor,
        noised_action: torch.Tensor,
        sigma: torch.Tensor,
        extra_inputs: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """Forward pass of the model.

        Args:

            state (torch.Tensor): State observation as condition for the model.
            noised_action (torch.Tensor): Noised action as input to the model.
            sigma (torch.Tensor): Sigma as condition for the model.
            extra_inputs (Dict[str, torch.Tensor]): Unused in this model.

        Returns:
            torch.Tensor: Model output. Shape (B, t_pred, action_size).
        """

        minibatch_size = noised_action.shape[0]

        # Process noised action
        action_embedding = self.action_embedding(noised_action)

        # Embed sigma
        # (B, 1, embedding_size)
        sigma_embedding = self.sigma_embedding(sigma.view(minibatch_size, -1)).unsqueeze(1)

        # Encoder
        # (B, t_obs, embedding_size)
        condition_embedding = self.condition_embedding(state)
        # (B, condition_time_steps, embedding_size)
        condition_embedding = torch.cat([sigma_embedding, condition_embedding], dim=1)

        x = self.drop(condition_embedding + self.condition_position_embedding)
        x = self.encoder(x)

        # (B, condition_time_steps, embedding_size)
        memory = x

        # Decoder
        token_embeddings = action_embedding
        # (B, t_pred, embedding_size)
        x = self.drop(token_embeddings + self.position_embedding)
        # (B, t_pred, embedding_size)
        x = self.decoder(tgt=x, memory=memory, tgt_mask=self.mask, memory_mask=self.memory_mask)

        # Head
        # The final step in the decoded sequence is passed to a final linear layer to create the ProDMP parameters
        x = self.ln_f(x[:, -1, :])
        params = self.head(x)

        # Set the latest prodmp parameters so that they can be used in predict_upsampled
        self.latest_prodmp_parameters = params

        # OPTIONAL TODO: Implement learnable tau and delay.
        trajectory = self.prodmp_handler.decode(params, **extra_inputs)

        return trajectory
