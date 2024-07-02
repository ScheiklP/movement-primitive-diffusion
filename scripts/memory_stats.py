import time
import hydra
import torch
import tabulate
import psutil

from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from typing import Tuple

from movement_primitive_diffusion.datasets.trajectory_dataset import SubsequenceTrajectoryDataset
from movement_primitive_diffusion.utils.data_utils import get_total_file_size, convert_size

OmegaConf.register_new_resolver("eval", eval)

CONFIG = "experiments/ligating_loop/train_prodmp_transformer.yaml"


def get_data_loaders(cfg: DictConfig) -> Tuple[DataLoader, DataLoader]:
    # Figure out which device to use
    if cfg.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        assert isinstance(cfg.device, str), f"Expected device to be a str, got {type(cfg.device)=}."
        assert cfg.device in ["cuda", "cpu"], f"Please set device to either cpu or cuda. Got {cfg.device=}."
        device = cfg.device
    cfg.device = device
    cfg.agent_config.device = device

    # Create the dataset and dataloaders
    dataset: SubsequenceTrajectoryDataset = hydra.utils.instantiate(cfg.dataset_config, _convert_="all")
    if cfg.dataset_fully_on_gpu:
        dataset.to(device)
    (train_dataset, val_dataset), _ = dataset.split([cfg.train_split, 1 - cfg.train_split])
    train_dataloader = DataLoader(train_dataset, **cfg.data_loader_config)
    val_dataloader = DataLoader(val_dataset, **cfg.data_loader_config)

    return train_dataloader, val_dataloader


def look_for_trajectory_dir(cfg: DictConfig) -> Path:
    relative_trajectory_dir = Path(__file__).parent / f"../data/{cfg.trajectory_dir}/"
    absolute_trajectory_dir = Path(cfg.trajectory_dir)
    if absolute_trajectory_dir.is_dir() and relative_trajectory_dir.is_dir():
        raise ValueError(f"Found two directories for trajectories: {relative_trajectory_dir=} and {absolute_trajectory_dir=}.")
    elif not absolute_trajectory_dir.is_dir() and not relative_trajectory_dir.is_dir():
        raise ValueError(f"Could not find trajectory directory. Looked in {relative_trajectory_dir=} and {absolute_trajectory_dir=}.")
    elif absolute_trajectory_dir.is_dir():
        trajectory_dir = absolute_trajectory_dir
    else:
        trajectory_dir = relative_trajectory_dir

    return trajectory_dir


@hydra.main(version_base=None, config_path="../conf", config_name=CONFIG)
def main(cfg: DictConfig) -> None:
    # Look for data
    trajectory_dir = look_for_trajectory_dir(cfg)
    cfg.dataset_config.trajectory_dirs = [path for path in trajectory_dir.iterdir() if path.is_dir()]

    # Get the names of the npz files that will actually be loaded
    npz_keys = cfg.dataset_config["keys"]
    state_datasize = get_total_file_size(trajectory_dir, npz_keys)
    if hasattr(cfg.dataset_config, "image_keys"):
        image_keys = cfg.dataset_config["image_keys"]
        image_datasize = get_total_file_size(trajectory_dir, image_keys)
        npz_keys = npz_keys + image_keys
    else:
        image_datasize = 0

    # Get the total size of the data
    total_data_size = get_total_file_size(trajectory_dir, npz_keys)

    # Get the currently available memory on RAM and GPU
    available_ram = psutil.virtual_memory().available
    available_gpu_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)

    # Get the dataloaders
    train_dataloader, val_dataloader = get_data_loaders(cfg)

    # Get the available memory after loading the data
    available_ram_after_loading = psutil.virtual_memory().available
    available_gpu_memory_after_loading = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)

    # Check how much time is required for one epoch of loading training and validation data
    start = time.perf_counter()
    for batch in train_dataloader:
        pass
    end = time.perf_counter()
    train_time = end - start

    start = time.perf_counter()
    for batch in val_dataloader:
        pass
    end = time.perf_counter()
    val_time = end - start

    header = ["", "Size"]
    table = [
        ["State", convert_size(state_datasize)],
        ["Images", convert_size(image_datasize)],
        ["Total", convert_size(total_data_size)],
        ["Number of Trajectories", len(cfg.dataset_config.trajectory_dirs)],
    ]
    print(tabulate.tabulate(table, headers=header, floatfmt=".2f", tablefmt="simple_outline"))

    header = ["Available", "Before Loading", "After Loading"]
    table = [
        ["RAM", convert_size(available_ram), convert_size(available_ram_after_loading)],
        ["GPU Memory", convert_size(available_gpu_memory), convert_size(available_gpu_memory_after_loading)],
    ]
    print(tabulate.tabulate(table, headers=header, floatfmt=".2f", tablefmt="simple_outline"))

    header = ["Option", "Value"]
    table = [
        ["device", cfg.device],
        ["train_split", cfg.train_split],
        ["dataset_fully_on_gpu", cfg.dataset_fully_on_gpu],
        ["pin_memory", cfg.data_loader_config.pin_memory],
        ["num_workers", cfg.data_loader_config.num_workers],
        ["batch_size", cfg.data_loader_config.batch_size],
    ]
    print(tabulate.tabulate(table, headers=header, floatfmt=".2f", tablefmt="simple_outline"))

    header = ["Split", "Data loader Time"]
    table = [
        ["Train Epoch", train_time],
        ["Val Epoch", val_time],
        ["Train Batch", train_time / len(train_dataloader)],
        ["Val Batch", val_time / len(val_dataloader)],
    ]
    print(tabulate.tabulate(table, headers=header, tablefmt="simple_outline"))


if __name__ == "__main__":
    main()
