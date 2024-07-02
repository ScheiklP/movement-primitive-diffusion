import torch
import hydra

from omegaconf import DictConfig
from typing import Optional, Tuple, Dict, List

from movement_primitive_diffusion.networks.sigma_embeddings import SIGMA_EMBEDDINGS


# Adapted from https://github.com/columbia-ai-robotics/diffusion_policy/blob/main/diffusion_policy/model/diffusion/transformer_for_diffusion.py
class CausalTransformer(torch.nn.Module):
    def __init__(
        self,
        action_size: int,
        state_size: int,
        sigma_embedding_config: DictConfig,
        t_pred: int,
        t_obs: int,
        predict_past: bool = False,
        n_layers: int = 8,
        n_heads: int = 4,
        embedding_size: int = 256,
        dropout_probability_embedding: float = 0.0,
        dropout_probability_attention: float = 0.01,
        n_cond_layers: int = 0,
    ) -> None:
        super().__init__()

        condition_time_steps = 1 + t_obs  # sigma + state
        action_time_steps = t_pred
        if predict_past:
            action_time_steps += t_obs - 1  # obs and act overlap at t0 -> t_obs - 1

        # Dropout layer for embedding
        self.drop = torch.nn.Dropout(dropout_probability_embedding)

        # Sigma embedding
        self.sigma_embedding: torch.nn.Module = hydra.utils.instantiate(sigma_embedding_config)
        if hasattr(self.sigma_embedding, "embedding_size"):
            if not self.sigma_embedding.embedding_size == embedding_size:
                raise ValueError(f"embedding_size of sigma_embedding ({self.sigma_embedding.embedding_size}) should be equal to embedding_size ({embedding_size})")

        # Linear layer for state embedding
        self.condition_embedding = torch.nn.Linear(state_size, embedding_size)

        # Linear layer for action embedding
        self.action_embedding = torch.nn.Linear(action_size, embedding_size)

        # Learnable vectors for positional encoding of action and state & sigma
        self.condition_position_embedding = torch.nn.Parameter(torch.zeros(1, condition_time_steps, embedding_size))
        self.position_embedding = torch.nn.Parameter(torch.zeros(1, action_time_steps, embedding_size))

        # Encoder
        if n_cond_layers > 0:
            encoder_layer = torch.nn.TransformerEncoderLayer(
                d_model=embedding_size,
                nhead=n_heads,
                dim_feedforward=4 * embedding_size,
                dropout=dropout_probability_attention,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = torch.nn.TransformerEncoder(
                encoder_layer=encoder_layer,
                num_layers=n_cond_layers,
            )
        else:
            self.encoder = torch.nn.Sequential(
                torch.nn.Linear(embedding_size, 4 * embedding_size),
                torch.nn.Mish(),
                torch.nn.Linear(4 * embedding_size, embedding_size),
            )

        # Decoder
        decoder_layer = torch.nn.TransformerDecoderLayer(
            d_model=embedding_size,
            nhead=n_heads,
            dim_feedforward=4 * embedding_size,
            dropout=dropout_probability_attention,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = torch.nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=n_layers,
        )

        # Attention mask
        # Causal mask to ensure that attention is only applied to the left in the input sequence.
        # TODO?: https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html#:~:text=attn_mask%20(Optional,types%20should%20match.
        # TODO?: For a binary mask, a True value indicates that the corresponding position is not allowed to attend.
        # torch.nn.Transformer uses additive mask as opposed to multiplicative mask in minGPT
        # therefore, the upper triangle should be -inf and others (including diag) should be 0.
        mask = (torch.triu(torch.ones(action_time_steps, action_time_steps)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
        self.register_buffer("mask", mask)

        if predict_past:
            t, s = torch.meshgrid(torch.arange(action_time_steps), torch.arange(condition_time_steps), indexing="ij")
            mask = t >= (s - 1)  # add one dimension since time is the first token in cond
            mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
            self.register_buffer("memory_mask", mask)
        else:
            self.register_buffer("memory_mask", None)

        # Decoder head
        self.ln_f = torch.nn.LayerNorm(embedding_size)
        self.head = torch.nn.Linear(embedding_size, action_size)

        # Init
        self.apply(self._init_weights)

    def _init_weights(self, module):
        ignore_types = (
            torch.nn.Dropout,
            torch.nn.TransformerEncoderLayer,
            torch.nn.TransformerDecoderLayer,
            torch.nn.TransformerEncoder,
            torch.nn.TransformerDecoder,
            torch.nn.ModuleList,
            torch.nn.Mish,
            torch.nn.Sequential,
        ) + SIGMA_EMBEDDINGS

        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, torch.nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.MultiheadAttention):
            weight_names = ["in_proj_weight", "q_proj_weight", "k_proj_weight", "v_proj_weight"]
            for name in weight_names:
                weight = getattr(module, name)
                if weight is not None:
                    torch.nn.init.normal_(weight, mean=0.0, std=0.02)

            bias_names = ["in_proj_bias", "bias_k", "bias_v"]
            for name in bias_names:
                bias = getattr(module, name)
                if bias is not None:
                    torch.nn.init.zeros_(bias)
        elif isinstance(module, torch.nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, CausalTransformer):
            torch.nn.init.normal_(module.position_embedding, mean=0.0, std=0.02)
            if module.condition_embedding is not None:
                torch.nn.init.normal_(module.condition_position_embedding, mean=0.0, std=0.02)
        elif isinstance(module, ignore_types):
            # no param
            pass
        else:
            raise RuntimeError("Unaccounted module {}".format(module))

    def get_optim_groups(self, weight_decay: float = 1e-3) -> List[dict]:
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.MultiheadAttention)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for module_name, module in self.named_modules():
            for parameter_name, _ in module.named_parameters():
                fpn = "%s.%s" % (module_name, parameter_name) if module_name else parameter_name  # full param name

                if parameter_name.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif parameter_name.startswith("bias"):
                    # MultiheadAttention bias starts with "bias"
                    no_decay.add(fpn)
                elif parameter_name.endswith("weight") and isinstance(module, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif parameter_name.endswith("weight") and isinstance(module, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add("position_embedding")
        if self.condition_position_embedding is not None:
            no_decay.add("condition_position_embedding")

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" % (str(param_dict.keys() - union_params),)

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        return optim_groups

    def configure_optimizers(self, learning_rate: float = 1e-4, weight_decay: float = 1e-3, betas: Tuple[float, float] = (0.9, 0.95), eps: float = 1e-08):
        optim_groups = self.get_optim_groups(weight_decay=weight_decay)
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, eps=eps)
        return optimizer

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
            torch.Tensor: Model output. Shape (B, action_time_steps, action_size).
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
        # (B, action_time_steps, embedding_size)
        x = self.drop(token_embeddings + self.position_embedding)
        # (B, action_time_steps, embedding_size)
        x = self.decoder(tgt=x, memory=memory, tgt_mask=self.mask, memory_mask=self.memory_mask)

        # head
        x = self.ln_f(x)
        x = self.head(x)
        # (B, action_time_steps, action_size)
        return x
