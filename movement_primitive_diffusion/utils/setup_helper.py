import os
import git
import hydra
import logging
import torch
import wandb

from copy import deepcopy
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from torch.utils.data import DataLoader
from typing import Tuple, List, Optional, Union

from movement_primitive_diffusion.agents.base_agent import BaseAgent
from movement_primitive_diffusion.datasets.process_batch import ProcessBatchProDMP
from movement_primitive_diffusion.datasets.trajectory_dataset import SubsequenceTrajectoryDataset
from movement_primitive_diffusion.workspaces.base_vector_workspace import BaseVectorWorkspace
from movement_primitive_diffusion.workspaces.base_workspace import BaseWorkspace
from movement_primitive_diffusion.utils.helper import tensor_to_list
from movement_primitive_diffusion.utils.mp_utils import ProDMPHandler

log = logging.getLogger(__name__)


def get_dataloaders_for_fixed_split(cfg: DictConfig) -> Tuple[DataLoader, DataLoader]:
    # Look up all available trajectory paths
    assert "train_trajectory_dir" in cfg and cfg.train_trajectory_dir is not None, "train_trajectory_dir must be set if fixed_split is True"
    assert "val_trajectory_dir" in cfg and cfg.val_trajectory_dir is not None, "val_trajectory_dir must be set if fixed_split is True"
    train_trajectory_dir = look_for_trajectory_dir(cfg.train_trajectory_dir)
    val_trajectory_dir = look_for_trajectory_dir(cfg.val_trajectory_dir)
    train_trajectories = [path for path in train_trajectory_dir.iterdir() if path.is_dir()]
    val_trajectories = [path for path in val_trajectory_dir.iterdir() if path.is_dir()]

    # Load all available trajectories to compute correct scaler values
    combined_trajectories = train_trajectories + val_trajectories
    combined_dataset_config = deepcopy(cfg.dataset_config)
    combined_dataset_config.trajectory_dirs = combined_trajectories
    combined_dataset = hydra.utils.instantiate(combined_dataset_config, _convert_="all")
    scaler_values = tensor_to_list(combined_dataset.scaler_values)

    # Delete the combined dataset to free up memory
    del combined_dataset

    # Set the scaler values in the dataset config
    cfg.dataset_config.scaler_values = scaler_values

    # Set the scaler values in the workspace config
    cfg.workspace_config.env_config.scaler_config.scaler_values = scaler_values

    # Instantiate train and validation datasets
    # If num_trajectories is set, only use the first num_trajectories for training
    train_dataset_config = deepcopy(cfg.dataset_config)
    val_dataset_config = deepcopy(cfg.dataset_config)
    train_dataset_config.trajectory_dirs = train_trajectories
    if cfg.get("num_trajectories", False):
        train_dataset_config.trajectory_dirs = train_dataset_config.trajectory_dirs[: cfg.num_trajectories]
    val_dataset_config.trajectory_dirs = val_trajectories

    # Training and validation data come from their own directories
    train_dataset = hydra.utils.instantiate(train_dataset_config, _convert_="all")
    val_dataset = hydra.utils.instantiate(val_dataset_config, _convert_="all")

    # Move the dataset to the correct device
    if cfg.dataset_fully_on_gpu:
        train_dataset.to(cfg.device)
        val_dataset.to(cfg.device)

    # If the batch size is -1, set it to the length of the dataset
    if cfg.data_loader_config.batch_size == -1:
        cfg.data_loader_config.batch_size = max(len(train_dataset), len(val_dataset))
        log.log(logging.INFO, f"Set batch size to {cfg.data_loader_config.batch_size} to fit all data in one batch.")

    train_dataloader = DataLoader(train_dataset, **cfg.data_loader_config)
    val_dataloader = DataLoader(val_dataset, **cfg.data_loader_config)

    return train_dataloader, val_dataloader


def get_dataloaders(cfg: DictConfig) -> Tuple[DataLoader, DataLoader]:
    # Look for data
    trajectory_dir = look_for_trajectory_dir(cfg.trajectory_dir)
    cfg.dataset_config.trajectory_dirs = [path for path in trajectory_dir.iterdir() if path.is_dir()]

    # If num_trajectories is set, only use the first num_trajectories
    if cfg.get("num_trajectories", False):
        cfg.dataset_config.trajectory_dirs = cfg.dataset_config.trajectory_dirs[: cfg.num_trajectories]

    # Create the dataset, move it to the correct device and split it into train and val data
    dataset: SubsequenceTrajectoryDataset = hydra.utils.instantiate(cfg.dataset_config, _convert_="all")
    if cfg.dataset_fully_on_gpu:
        dataset.to(cfg.device)
    (train_dataset, val_dataset), _ = dataset.split([cfg.train_split, 1 - cfg.train_split])

    # Set the scaler values in the workspace config
    cfg.workspace_config.env_config.scaler_config.scaler_values = tensor_to_list(dataset.scaler_values)

    # Delete the original dataset to free up memory immediately
    del dataset

    # If the batch size is -1, set it to the length of the dataset
    if cfg.data_loader_config.batch_size == -1:
        cfg.data_loader_config.batch_size = max(len(train_dataset), len(val_dataset))
        log.log(logging.INFO, f"Set batch size to {cfg.data_loader_config.batch_size} to fit all data in one batch.")

    train_dataloader = DataLoader(train_dataset, **cfg.data_loader_config)
    val_dataloader = DataLoader(val_dataset, **cfg.data_loader_config)

    return train_dataloader, val_dataloader


def setup_agent_and_workspace(cfg: DictConfig) -> Tuple[BaseAgent, BaseWorkspace]:
    """Helper function to update the hydra config and instantiate agent and workspace.

    Args:
        cfg (DictConfig): The hydra config.

    Returns:
        agent (BaseAgent): The agent.
        workspace (BaseWorkspace): The workspace.
    """

    # Figure out which device to use
    if cfg.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        assert isinstance(cfg.device, str), f"Expected device to be a str, got {type(cfg.device)=}."
        assert cfg.device in ["cuda", "cpu"], f"Please set device to either cpu or cuda. Got {cfg.device=}."
        device = cfg.device
    cfg.device = device
    cfg.agent_config.device = device

    # Instantiate the agent
    agent: BaseAgent = hydra.utils.instantiate(cfg.agent_config)

    # Make sure sigma_data is set if the scaling needs it
    if scaling := getattr(agent.model, "scaling", False):
        if getattr(scaling, "sigma_data", False) is None:
            raise ValueError("Please set sigma_data in the scaling module of the model.")

    # Instantiate the workspace
    workspace: BaseWorkspace = hydra.utils.instantiate(cfg.workspace_config)

    return agent, workspace


def setup_train(cfg: DictConfig) -> Tuple[DataLoader, DataLoader, BaseAgent, Union[BaseWorkspace, BaseVectorWorkspace]]:
    """Helper function to update the hydra config and instantiate the dataloaders, agent, and workspace.

    Args:
        cfg (DictConfig): The hydra config.

    Returns:
        train_dataloader (DataLoader): The dataloader for the training data.
        val_dataloader (DataLoader): The dataloader for the validation data.
        agent (BaseAgent): The agent.
        workspace (BaseWorkspace): The workspace.
    """
    # Deactivate tqdm if configured
    if cfg.get("deactivate_tqdm", False):
        os.environ["TQDM_DISABLE"] = "1"

    # Figure out which device to use
    if cfg.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        assert isinstance(cfg.device, str), f"Expected device to be a str, got {type(cfg.device)=}."
        assert cfg.device in ["cuda", "cpu"], f"Please set device to either cpu or cuda. Got {cfg.device=}."
        device = cfg.device
    cfg.device = device
    cfg.agent_config.device = device

    # Depending on cfg.fixed_split, either load all the data from cfg.trajectory_dir and split it with the split ratio
    # or load the data from cfg.train_trajectory_dir and cfg.val_trajectory_dir
    if "fixed_split" in cfg and cfg.fixed_split:
        train_dataloader, val_dataloader = get_dataloaders_for_fixed_split(cfg)
    else:
        train_dataloader, val_dataloader = get_dataloaders(cfg)

    # Get a batch of data to determine the observation sizes and validate the observation keys
    data = next(iter(train_dataloader))

    # Set the process_batch observation keys based on the encoder config
    encoder_observation_keys = []
    for network_config in cfg.agent_config.encoder_config.network_configs:
        encoder_observation_keys.append(network_config.observation_key)
    cfg.agent_config.process_batch_config.observation_keys = encoder_observation_keys

    # VALIDATION: Check that these keys are present in the data
    for key in encoder_observation_keys:
        assert key in data.keys(), f"Key {key} not present in data"

    # Set the observation sizes in the encoder config
    for network_config in cfg.agent_config.encoder_config.network_configs:
        network_config.feature_size = list(data[network_config.observation_key].shape[2:])
        if hasattr(inner_config := network_config.network_config, "feature_size"):
            inner_config.feature_size = network_config.feature_size

    # Set the sizes in the process_batch config
    for info in cfg.agent_config.process_batch_config.action_keys:
        info.feature_size = list(data[info.key].shape[2:])

    # NOTE: movement_primitive_diffusion.utils.lr_scheduler.get_scheduler expects the number of training steps as argument.
    # to not break compatibility with directly instantiating other schedulers, we check for the existence of the
    # num_training_steps attribute.
    if hasattr(cfg.agent_config.lr_scheduler_config, "num_training_steps"):
        # Figure out the number of training steps for the LR scheduler
        if cfg.epochs is None:
            raise ValueError("If you want to use an lr scheduler wit num_training_steps, you need to specify the number of epochs.")
        cfg.agent_config.lr_scheduler_config.num_training_steps = len(train_dataloader) * cfg.epochs

    # Instantiate the agent
    agent: BaseAgent = hydra.utils.instantiate(cfg.agent_config)

    # If necessary (there is a sigma_data, and its value is None), calculate and set sigma_data for scaling
    if scaling := getattr(agent.model, "scaling", False):
        if getattr(scaling, "sigma_data", False) is None:
            scaling.set_sigma_data(scaling.calculate_sigma_data_of_action(agent, train_dataloader))

    # Instantiate the workspace
    workspace: BaseWorkspace = hydra.utils.instantiate(cfg.workspace_config)

    return train_dataloader, val_dataloader, agent, workspace


def setup_prodmp_optim(cfg: DictConfig) -> Tuple[DataLoader, ProcessBatchProDMP, ProDMPHandler]:
    # Figure out which device to use
    if cfg.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        assert isinstance(cfg.device, str), f"Expected device to be a str, got {type(cfg.device)=}."
        assert cfg.device in ["cuda", "cpu"], f"Please set device to either cpu or cuda. Got {cfg.device=}."
        device = cfg.device
    cfg.device = device

    # Create the dataset and dataloader
    dataset: SubsequenceTrajectoryDataset = hydra.utils.instantiate(cfg.dataset_config, _convert_="all")
    if cfg.dataset_fully_on_gpu:
        dataset.to(device)
    if cfg.data_loader_config.batch_size == -1:
        cfg.data_loader_config.batch_size = len(dataset)
    dataloader = DataLoader(dataset, **cfg.data_loader_config)

    # Create the process batch function
    process_batch_function: ProcessBatchProDMP = hydra.utils.instantiate(cfg.process_batch_config)

    # Create the ProDMP handler
    prodmp_handler: ProDMPHandler = hydra.utils.instantiate(cfg.prodmp_handler_config)

    return dataloader, process_batch_function, prodmp_handler


def look_for_trajectory_dir(search_dir: str) -> Path:
    # Get the root directory of the repository
    git_repo = git.Repo(".", search_parent_directories=True)
    git_root = git_repo.working_tree_dir

    relative_trajectory_dir = Path(f"{git_root}/data/{search_dir}/")
    absolute_trajectory_dir = Path(search_dir)

    if absolute_trajectory_dir.is_dir() and relative_trajectory_dir.is_dir():
        raise ValueError(f"Found two directories for trajectories: {relative_trajectory_dir=} and {absolute_trajectory_dir=}.")
    elif not absolute_trajectory_dir.is_dir() and not relative_trajectory_dir.is_dir():
        raise ValueError(f"Could not find trajectory directory. Looked in {relative_trajectory_dir=} and {absolute_trajectory_dir=}.")
    elif absolute_trajectory_dir.is_dir():
        trajectory_dir = absolute_trajectory_dir
    else:
        trajectory_dir = relative_trajectory_dir

    return trajectory_dir


def setup_wandb_metrics(workspace_result_keys: List[str], performance_metric: str) -> None:
    """Helper function to set up the metrics for wandb.

    Note:
        We define this function to log the metrics over epochs.

    Args:
        workspace_result_keys (List[str]): The keys of the workspace results of workspace.test_agent(agent).
        performance_metric (str): The metric that is used to determine whether a model is better than the previous best.
    """

    wandb.define_metric("epoch")
    metric_names = ["lr", "loss", "val_loss", "start_point_deviation", "end_point_deviation", "best_epoch", f"best_{performance_metric}"] + workspace_result_keys
    for metric_name in metric_names:
        wandb.define_metric(metric_name, step_metric="epoch")


def setup_wandb_test_metrics(workspace_result_keys: List[str]) -> None:
    """Helper function to set up the metrics for wandb in the test scripts.

    Note:
        We define this function to log the metrics over epochs.

    Args:
        workspace_result_keys (List[str]): The keys of the workspace results of workspace.test_agent(agent).
    """

    wandb.define_metric("epoch")
    for metric_name in workspace_result_keys:
        wandb.define_metric(metric_name, step_metric="epoch")


def get_group_from_override(length: int = 2, ignore_keys: Optional[List[str]] = None) -> str:
    """Helper function to get the group name from the hydra overrides.

    Basically we take the last part of the override, split it by '_', and then take the first letter of each part.
    ==> agent_config.model_config.train_btm_image_prodmp_residual_mlp -> t_b_i_p_r_m (length=1) or tr_bi_im_pr_ml (length=2)

    Returns:
        str: The group name.
    """
    overrides = hydra.utils.HydraConfig.get()["overrides"]["task"]
    overrides_shortened = []
    ignore_keys = ignore_keys or []
    for override in overrides:
        if "seed" in override:
            continue
        override_value = override.split("=")[-1]
        override_key = override.split("=")[-2]
        key_components = override_key.split(".")  # overrides from setting a param.value: val
        if len(key_components) == 1:
            key_components = key_components[0].split("/")  # overrides from overriding a config/value: config_name

        if key_components[-1] in ignore_keys:
            continue
        else:
            override_shortened = "_".join([o[:length] for o in key_components[-1].split("_")])
            overrides_shortened.append(f"{override_shortened}={override_value}")

    return ",".join(overrides_shortened)


def parse_wandb_to_hydra_config(wandb_config: DictConfig) -> DictConfig:
    """Parses a wandb config to a hydra config.

    Args:
        wandb_config (DictConfig): The wandb config. At the first level, a wandb config
        contains a key "wandb_version" and a key for each hyperparameter. Each hyperparameter
        contains a description and a value. We only need the value.

    Returns:
        hydra_config (DictConfig): The hydra config. Removes the "wandb_version" key and
        only keeps the hyperparameters and their values.
    """
    hydra_config = OmegaConf.create()

    for key, value in wandb_config.items():
        if key == "wandb_version":
            continue

        if "desc" in value and "value" in value:
            hydra_config[key] = value["value"]
        else:
            raise ValueError(f"Expected {key} to have a value and a description, got {value=}")

    return hydra_config
