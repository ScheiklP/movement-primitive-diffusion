from typing import Tuple, Dict

import hydra
import torch
from omegaconf import DictConfig

from robomimic.algo import algo_factory
from robomimic.algo.algo import PolicyAlgo
from robomimic.config import config_factory
import robomimic.scripts.generate_paper_configs as gpc
from robomimic.scripts.generate_paper_configs import (
    modify_config_for_default_image_exp,
    modify_config_for_default_low_dim_exp,
    modify_config_for_dataset,
)

from movement_primitive_diffusion.models.base_model import BaseModel

def get_robomimic_config(
        algo_name='bc_rnn',
        hdf5_type='low_dim',
        task_name='square',
        dataset_type='ph'
    ):
    base_dataset_dir = '/tmp/null'
    filter_key = None

    # decide whether to use low-dim or image training defaults
    modifier_for_obs = modify_config_for_default_image_exp
    if hdf5_type in ["low_dim", "low_dim_sparse", "low_dim_dense"]:
        modifier_for_obs = modify_config_for_default_low_dim_exp

    algo_config_name = "bc" if algo_name == "bc_rnn" else algo_name
    config = config_factory(algo_name=algo_config_name)
    # turn into default config for observation modalities (e.g.: low-dim or rgb)
    config = modifier_for_obs(config)
    # add in config based on the dataset
    config = modify_config_for_dataset(
        config=config,
        task_name=task_name,
        dataset_type=dataset_type,
        hdf5_type=hdf5_type,
        base_dataset_dir=base_dataset_dir,
        filter_key=filter_key,
    )
    # add in algo hypers based on dataset
    algo_config_modifier = getattr(gpc, f'modify_{algo_name}_config_for_dataset')
    config = algo_config_modifier(
        config=config,
        task_name=task_name,
        dataset_type=dataset_type,
        hdf5_type=hdf5_type,
    )
    return config
class RobomimicLSTMGMM(BaseModel):
    def __init__(
        self,
        inner_model_config: DictConfig,
    ):
        super().__init__()

        self.inner_model = hydra.utils.instantiate(inner_model_config)

    def loss(self, state: Tuple[torch.Tensor, torch.Tensor], action: torch.Tensor, extra_inputs: Dict, return_deviations: bool = False) -> torch.Tensor:

        predicted_trajectory, trajectory_uncertainty = self.forward(state, extra_inputs, get_uncertainty=True)

        # change the axis such that value shape and pairs are switched (B, combinations, value shape, 2) -> swith time and value axis
        predicted_trajectory = torch.einsum("...ji->...ij", predicted_trajectory)
        trajectory_uncertainty = torch.einsum("...ji->...ij", trajectory_uncertainty)

        # calculate loss
        loss = nll_loss(action, predicted_trajectory, trajectory_uncertainty)

        if return_deviations:
            return loss, predicted_trajectory, trajectory_uncertainty
        else:
            return loss

    def forward(self, state: Tuple[torch.Tensor, torch.Tensor], extra_inputs: Dict, get_uncertainty: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            state: Tuple of (state, state_uncertainty)
            extra_inputs: Dict of extra inputs
            get_uncertainty: Whether to return uncertainty

        Returns:
            predicted_trajectory: Predicted trajectory
            trajectory_uncertainty: Uncertainty of predicted trajectory
        """
        # Get state and state uncertainty
        state, state_uncertainty = state

        # Get extra inputs