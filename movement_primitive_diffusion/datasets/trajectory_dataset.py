import numpy as np
import pickle
import re
import torch
import logging

from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import read_image

from copy import deepcopy
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Union, Any

from movement_primitive_diffusion.datasets.scalers import normalize, standardize, get_scaler_values

log = logging.getLogger(__name__)


def read_numpy_file(path: Union[Path, str]) -> Union[Dict[str, Any], np.ndarray]:
    """Read numpy file

    Args:
        path (Union[Path, str]): path to numpy file

    Returns:
        data (Dict[str, Any]): data stored in numpy file
    """

    if not Path(path).exists():
        raise FileNotFoundError(f"File {path} does not exist.")

    if Path(path).suffix == ".npy":
        data = np.load(path, allow_pickle=True).item()
        if not isinstance(data, dict):
            raise TypeError(f"File {path} does not contain a dictionary. Only dictionaries are supported as we assume that uncompressed files contain a trajectory dictionary.")
        return data

    elif Path(path).suffix == ".npz":
        data = np.load(path, allow_pickle=True)
        if len(data.files) != 1:
            raise ValueError(f"File {path} contains more than one numpy array. Only one is supported as we assume that compressed files contain one array per file.")
        data = data[data.files[0]]
        return data

    else:
        raise ValueError(f"File {path} has an unknown file extension. Only .npy and .npz are supported.")


class TrajectoryDataset(Dataset):
    """Dataset class for loading trajectories.

    Args:
        trajectory_dirs (List[Path]): List of paths to directories containing trajectories. Each directory should contain a single trajectory. The trajectory should be stored as a numpy file with the keys specified in keys.
        keys (List[str]): List of keys to load from the trajectory files.
        dt (float): Time step of the trajectories.
        target_dt (Optional[float]): Target time step of the trajectories. If None, the trajectories are not resampled. Defaults to None.
        normalize_keys (Union[bool, List[str]]): Whether to normalize the keys. If True, all keys are normalized. If a list of keys is provided, only the keys in the list are normalized. Defaults to False.
        normalize_symmetrically (bool): Whether to normalize to [-1, 1] (True) or [0, 1] (False). Defaults to False.
        standardize_keys (Union[bool, List[str]]): Whether to standardize the keys. If True, all keys are standardized. If a list of keys is provided, only the keys in the list are standardized. Defaults to False.
        scaler_values (Optional[Dict[str, Dict[str, torch.Tensor]]]): Dictionary of normalizer values. If None, the normalizer values are calculated from the data. Defaults to None.
        image_keys (Optional[List[str]]): List of keys to load images from. Defaults to None. These images can be loaded from a directory of png files or a single .npz file.
        image_sizes (Optional[List[Tuple[int, int]]]): List of sizes of the images to load. If None, the images are loaded in their original size. Defaults to None.
        crop_sizes (Optional[List[Tuple[int, int]]]): List of sizes of the images to crop to. If None, the images are not cropped. Defaults to None.
        random_crop (Union[List[bool], bool]): Whether to randomly crop the images or crop them from the center. If True, all images are randomly cropped. If a list of booleans is provided, only the images with a corresponding True value are randomly cropped. Defaults to True.
        normalize_images (Union[List[bool], bool]): Whether to normalize the images. If True, all images are normalized. If a list of booleans is provided, only the images with a corresponding True value are normalized. Defaults to True.
        calculate_velocities_from_to (List[Tuple[str, str]]): List of tuples of keys to calculate velocities from and to. Defaults to [].
        recalculate_velocities_from_to (List[Tuple[str, str]]): List of tuples of keys to recalculate velocities after normalization of position values. Defaults to [].
        pad_start (int): Number of steps to pad at the start of each trajectory. Defaults to 0.
        pad_end (int): Number of steps to pad at the end of each trajectory. Defaults to 0.

    TODO:
        - Add support for more image transforms. E.g. normalization, RGB2GRAY, etc
        - Add support for per key image transforms. E.g. only normalize the images of a specific key.
    """

    def __init__(
        self,
        trajectory_dirs: List[Path],
        keys: List[str],
        dt: float,
        target_dt: Optional[float] = None,
        normalize_keys: Union[bool, List[str]] = False,
        normalize_symmetrically: bool = False,
        standardize_keys: Union[bool, List[str]] = False,
        scaler_values: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
        image_keys: Optional[List[str]] = None,
        image_sizes: Optional[List[Tuple[int, int]]] = None,  # HxW
        crop_sizes: Optional[List[Tuple[int, int]]] = None,  # HxW
        random_crop: Union[List[bool], bool] = True,
        normalize_images: Union[List[bool], bool] = True,
        calculate_velocities_from_to: List[Tuple[str, str]] = [],
        recalculate_velocities_from_to: List[Tuple[str, str]] = [],
        pad_start: int = 0,
        pad_end: int = 0,
    ):
        self.trajectory_dirs = trajectory_dirs
        self.keys = keys
        self.image_keys = image_keys if image_keys is not None else []
        self.pad_start = pad_start
        self.pad_end = pad_end

        # Checks normalize_keys and standardize_keys against self.keys and returns a dictionary with the validated values.
        validated_values = self.__check_keys(normalize_keys, standardize_keys)
        self.normalize_keys = validated_values["normalize_keys"]
        self.standardize_keys = validated_values["standardize_keys"]

        # Whether to scale to [-1, 1] instead of [0, 1]
        self.normalize_symmetrically = normalize_symmetrically

        # Load trajectories into self.trajectories as a list of dictionaries -> [trajectory_1, trajectory_2, ...], where each trajectory is a dictionary -> {"key_1": torch.Tensor, "key_2": torch.Tensor, ...}
        self.__load_trajectories()

        # Check scaler_values against self.normalize_keys and self.standardize_keys and returns a dictionary with the validated values.
        self.scaler_values = self.__check_scaler_values(scaler_values, self.normalize_keys, self.standardize_keys)

        # Set initial device value based on loaded data
        trajectory = self.trajectories[0]
        self.device = trajectory[list(trajectory.keys())[0]].device

        # Optionally subsample trajectory
        if target_dt is not None:
            self.__subsample_time_steps(dt, target_dt)
            self.dt = target_dt
        else:
            self.dt = dt

        # Calculate velocities from positions
        for position_key, velocity_key in calculate_velocities_from_to:
            self.__calculate_velocity(position_key, velocity_key)

        # Normalize and standardize data
        self.__normalize_data(self.normalize_keys, self.scaler_values)
        self.__standardize_data(self.standardize_keys, self.scaler_values)

        # Recalculate velocities from normalized positions
        for position_key, velocity_key in recalculate_velocities_from_to:
            if position_key not in self.normalize_keys:
                raise RuntimeWarning(f"Received position_key {position_key} for recalculation of velocities, but this key is not in normalize_keys.")
            if position_key in self.standardize_keys:
                raise RuntimeError(f"Received position_key {position_key} for recalculation of velocities, but this key is also in standardize_keys. This is not supported.")
            self.__calculate_velocity(position_key, velocity_key)

        # Compose image transform
        self.image_transform = {}
        self.image_shape = {}
        random_crop = [random_crop] * len(self.image_keys) if isinstance(random_crop, bool) else random_crop
        normalize_images = [normalize_images] * len(self.image_keys) if isinstance(normalize_images, bool) else normalize_images
        image_sizes = [None] * len(self.image_keys) if image_sizes is None else image_sizes
        crop_sizes = [None] * len(self.image_keys) if crop_sizes is None else crop_sizes
        if len(self.image_keys):
            assert len(self.image_keys) == len(image_sizes) == len(crop_sizes) == len(random_crop) == len(normalize_images), "The number of image keys, image sizes, crop sizes, random crop values and normalize image values must be the same."
        for index, key in enumerate(self.image_keys):
            self.image_transform[key], self.image_shape[key] = self.__compose_image_transform(
                key,
                image_sizes[index],
                crop_sizes[index],
                random_crop[index],
                normalize_images[index],
            )

    def to(self, device: Union[str, torch.device]) -> None:
        """Moves all data to the given device.

        Args:
            device (torch.device): Device to move the data to.
        """

        for trajectory in self.trajectories:
            for key in trajectory.keys():
                if isinstance(trajectory[key], torch.Tensor):
                    trajectory[key] = trajectory[key].to(device)
                else:
                    # If the value is not a tensor, it is a list of image paths
                    assert isinstance(trajectory[key], list), f"Value for key {key} is not a tensor or a list."
                    assert isinstance(trajectory[key][0], Path), f"Value for key {key} is not a list of paths."
        self.device = device

    def split(self, split_ratios: List[float], shuffle: bool = True) -> Tuple[List["TrajectoryDataset"], torch.Tensor]:
        """Splits the dataset into multiple datasets.

        Args:
            split_ratios (List[float]): List of fractions to split the dataset into. The fractions must sum to one.
            shuffle (bool, optional): Whether to shuffle the dataset before splitting. Defaults to True.

        Raises:
            ValueError: If the fractions do not sum to one.
            ValueError: If any of the fractions are too small.

        Returns:
            Tuple[List[TrajectoryDataset], torch.Tensor]: List of datasets and a tensor of indices for each dataset.
        """

        # Ensure the fractions sum to one
        if np.sum(split_ratios) != 1.0:
            raise ValueError("Fractions must sum to one.")

        # Create an array of indices
        trajectory_indices = np.arange(len(self.trajectories))

        # Shuffle the indices if requested
        if shuffle:
            np.random.shuffle(trajectory_indices)

        # Convert fractions to index counts
        fractions_as_counts = [int(len(trajectory_indices) * fraction) for fraction in split_ratios]

        # Adjust last count to capture any rounding issues
        fractions_as_counts[-1] = len(trajectory_indices) - np.sum(fractions_as_counts[:-1])

        # Ensure that each count is at least one
        if any([count < 1 for count in fractions_as_counts]):
            raise ValueError("Fractions are too small.")

        # Split the indices based on counts
        split_indices = np.split(trajectory_indices, np.cumsum(fractions_as_counts[:-1]))

        # Create a dataset for each split by copying the current dataset and adjusting the trajectories and trajectory_lengths
        split_datasets = []
        for split in split_indices:
            dataset = deepcopy(self)
            dataset.trajectory_dirs = [dataset.trajectory_dirs[index] for index in split]
            dataset.trajectories = [dataset.trajectories[index] for index in split]
            dataset.trajectory_lengths = [dataset.trajectory_lengths[index] for index in split]
            split_datasets.append(dataset)

        return split_datasets, split_indices

    def __compose_image_transform(self, image_key: str, image_size: Optional[List[int]], crop_size: Optional[List[int]], random_crop: bool, normalize: bool) -> Tuple[transforms.Compose, Tuple[int, int, int]]:
        image_transform_list = []

        if normalize:
            # Divide by 255 to scale to [0, 1]
            image_transform_list.append(transforms.Lambda(lambda x: x / 255.0))

        # Handle preloaded images and image paths
        if isinstance(sample_image_path := self.trajectories[0][image_key][0], Path):
            # Reads image from path as uint8 and with shape (C, H, W)
            image = read_image(str(sample_image_path.absolute()))
            # Determine original image shape
            org_image_shape = tuple(image.shape)
        else:
            # Get tensor image
            tensor_image = self.trajectories[0][image_key][0]
            assert isinstance(tensor_image, torch.Tensor), f"Image must be a tensor, but is {type(tensor_image)}."
            # Determine original image shape
            org_image_shape = tuple(tensor_image.shape)
            assert len(org_image_shape) == 3, f"Image must have 3 dimensions, but has {len(org_image_shape)}."
            assert org_image_shape[0] == 3, f"Image must have 3 channels, but has {org_image_shape[0]} in shape {org_image_shape}."

        # Resizing images
        if image_size is None:
            image_shape = org_image_shape
        else:
            assert isinstance(image_size, list) and len(image_size) == 2
            image_shape = tuple([3] + image_size)
            if image_size != list(org_image_shape[1:]):
                image_transform_list.append(transforms.Resize(image_size, antialias=True))

        # (Random) cropping images
        if crop_size is not None:
            if not crop_size[0] <= image_shape[1] and crop_size[1] <= image_shape[2]:
                raise ValueError(f"Crop size {crop_size} is larger than image size {image_shape[1:]}.")
            image_shape = tuple([3] + crop_size)
            if random_crop:
                image_transform_list.append(transforms.RandomCrop(crop_size))
            else:
                image_transform_list.append(transforms.CenterCrop(crop_size))

        return transforms.Compose(image_transform_list), image_shape

    def __load_trajectories(self) -> None:
        # Load numpy arrays for all trajectories
        self.trajectories = []
        self.trajectory_lengths = []

        for trajectory_dir in self.trajectory_dirs:
            trajectory = {}
            # Load the specified keys into torch tensors
            for key in self.keys:
                trajectory_tensor = torch.tensor(read_numpy_file(trajectory_dir / Path(key + ".npz")), dtype=torch.float32)
                trajectory[key] = self._pad_tensors(trajectory_tensor)

            # For image keys, load either the npz file or keep a list of image paths
            for key in self.image_keys:
                # Check if the key is saved as a npz file, or as separate png files in a directory
                # NOTE: The directory takes priority over the npz file
                is_a_npz_file = (trajectory_dir / Path(key + ".npz")).exists()
                is_a_directory = (trajectory_dir / Path(key)).is_dir()

                if is_a_directory:
                    log.log(logging.INFO, f"Loading image files from directory {trajectory_dir / Path(key)} during data loading.")
                    # Keep a list of all image paths for each trajectory
                    trajectory[key] = [image_path for image_path in (trajectory_dir / Path(key)).iterdir() if image_path.suffix == ".png"]
                    # Sort image paths by name based on the last sequence of digits in the name
                    trajectory[key].sort(key=lambda image_path: int(re.findall(r"\d+", image_path.stem)[-1]))
                    trajectory[key] = self._pad_list(trajectory[key])
                elif is_a_npz_file:
                    # Load the npz file
                    images_tensor = torch.tensor(read_numpy_file(trajectory_dir / Path(key + ".npz")), dtype=torch.float32)
                    trajectory[key] = self._pad_tensors(images_tensor)
                else:
                    raise ValueError(f"Could not find '{key}.npz' or '{key}' directory in '{trajectory_dir}'.")

            self.trajectories.append(trajectory)

        # Validate that all contents of the trajectories have the same length and save the lengths.
        for trajectory in self.trajectories:
            for key in trajectory.keys():
                if len(trajectory[key]) != len(trajectory[list(trajectory.keys())[0]]):
                    raise RuntimeError(f"Trajectory {trajectory} has different lengths for {key} and {list(trajectory.keys())[0]}.")

            self.trajectory_lengths.append(len(trajectory[list(trajectory.keys())[0]]))

    def __subsample_time_steps(self, dt: float, target_dt: float):
        if dt > target_dt:
            raise ValueError(f"dt ({dt}) must be smaller than target_dt ({target_dt}) for trajectory subsampling.")

        for trajectory_index, trajectory in enumerate(self.trajectories):
            subsample_indices = torch.arange(0, self.trajectory_lengths[trajectory_index], int(min(target_dt / dt, self.trajectory_lengths[trajectory_index] - 1)))
            self.trajectory_lengths[trajectory_index] = len(subsample_indices)
            for key in trajectory.keys():
                trajectory[key] = trajectory[key][subsample_indices]

    def __normalize_data(self, normalize_keys: List[str], scaler_values: Dict[str, torch.Tensor]) -> None:
        # Data normalization
        for key in normalize_keys:
            # Save normalizer values for normalization and denormalization
            if scaler_values is None:
                self.scaler_values[key] = get_scaler_values(torch.cat([trajectory[key] for trajectory in self.trajectories]))
            # Normalize all trajectories
            for trajectory in self.trajectories:
                trajectory[key] = normalize(trajectory[key], self.scaler_values[key], symmetric=self.normalize_symmetrically)

    def __standardize_data(self, standardize_keys: List[str], scaler_values: Dict[str, torch.Tensor]) -> None:
        # Data standardization
        for key in standardize_keys:
            # Save normalizer values for standardization and destandardization
            if scaler_values is None:
                self.scaler_values[key] = get_scaler_values(torch.cat([trajectory[key] for trajectory in self.trajectories]))
            # Standardize all trajectories
            for trajectory in self.trajectories:
                trajectory[key] = standardize(trajectory[key], self.scaler_values[key])

    def __check_keys(self, normalize_keys: Union[List[str], bool], standardize_keys: Union[List[str], bool]) -> Dict[str, List[str]]:
        # Check normalizer keys
        if isinstance(normalize_keys, list):
            for key in normalize_keys:
                if key not in self.keys:
                    raise ValueError(f"normalize_keys contains {key} which is not in keys.")
            validated_normalize_keys = normalize_keys
        elif isinstance(normalize_keys, bool):
            if normalize_keys:
                validated_normalize_keys = self.keys.copy()
            else:
                validated_normalize_keys = []
        else:
            raise ValueError(f"normalize_keys must be bool or list of strings, but is {type(normalize_keys)}.")

        # Check standardizer keys
        if isinstance(standardize_keys, list):
            for key in standardize_keys:
                if key not in self.keys:
                    raise ValueError(f"standardize_keys contains {key} which is not in keys.")
                if key in validated_normalize_keys:
                    raise ValueError(f"standardize_keys contains {key} which is also in normalize_keys.")
            validated_standardize_keys = standardize_keys
        elif isinstance(standardize_keys, bool):
            if standardize_keys:
                validated_standardize_keys = self.keys.copy()
            else:
                validated_standardize_keys = []
        else:
            raise ValueError(f"standardize_keys must be bool or list of strings, but is {type(standardize_keys)}.")

        # Check if all keys are present in all trajectory dirs
        for trajectory_dir in self.trajectory_dirs:
            for key in self.keys:
                if not (trajectory_dir / Path(key + ".npz")).is_file():
                    raise RuntimeError(f"Could not find {key}.npz in {trajectory_dir}.")

            # Check if all trajectory dirs have rgb image dirs
            # if self.index_rgb_dir:
            #     for key in self.rgb_keys:
            #         if not (trajectory_dir / Path(key)).is_dir():
            #             raise RuntimeError(f"Could not find {key} dir in {trajectory_dir}.")

        return {
            "standardize_keys": validated_standardize_keys,
            "normalize_keys": validated_normalize_keys,
        }

    def __check_scaler_values(self, scaler_values: Union[Dict[str, torch.Tensor], None], normalize_keys: List[str], standardize_keys: List[str]) -> Dict[str, torch.Tensor]:
        # Check if scaler_values is valid
        if scaler_values is not None:
            for key in normalize_keys + standardize_keys:
                if key not in scaler_values:
                    raise ValueError(f"scaler_values does not contain {key}.")
                else:
                    # Convert contents to torch tensors
                    for statistics_key, statistics_value in scaler_values[key].items():
                        if isinstance(statistics_value, torch.Tensor):
                            scaler_values[key][statistics_key] = statistics_value.type(torch.float32)
                        elif statistics_value is None:
                            pass
                        elif isinstance(statistics_value, np.ndarray):
                            scaler_values[key][statistics_key] = torch.from_numpy(statistics_value).type(torch.float32)
                        elif isinstance(statistics_value, list):
                            scaler_values[key][statistics_key] = torch.tensor(statistics_value).type(torch.float32)
                        elif isinstance(statistics_value, float):
                            scaler_values[key][statistics_key] = torch.tensor(statistics_value).type(torch.float32)
                        else:
                            raise TypeError(f"scaler_values[{key}][{statistics_key}] is of type {type(statistics_value)}, but must be torch.Tensor, np.array, or list.")

            validated_scaler_values = scaler_values
        else:
            # If the scaler values are not provided, we will calculate them from the data
            validated_scaler_values = {key: {} for key in normalize_keys + standardize_keys}

        # Only calculate values for mean and std if they are not already provided
        for key in standardize_keys:
            relevant_keys = ["mean", "std"]
            if not all([relevant_key in validated_scaler_values[key] and validated_scaler_values[key][relevant_key] is not None for relevant_key in relevant_keys]):
                new_scaler_values = get_scaler_values(torch.cat([trajectory[key] for trajectory in self.trajectories]))
                # NOTE: If all the values are the same, std is 0, and standardization would be invalid: (val - mean) / std -> 0/0
                # We thus set the std value to 1, so the standardized value is 0. The same works for destandardization:  val (<- 0) * std + mean -> mean
                new_scaler_values["std"][new_scaler_values["std"] == 0.0] = 1.0
                for relevant_key in relevant_keys:
                    if not relevant_key in validated_scaler_values[key] or validated_scaler_values[key][relevant_key] is None:
                        validated_scaler_values[key][relevant_key] = new_scaler_values[relevant_key]

        # Only calculate values for min and max if they are not already provided
        for key in normalize_keys:
            relevant_keys = ["min", "max"]
            if not all([relevant_key in validated_scaler_values[key] and validated_scaler_values[key][relevant_key] is not None for relevant_key in relevant_keys]):
                new_scaler_values = get_scaler_values(torch.cat([trajectory[key] for trajectory in self.trajectories]))
                # NOTE: If min and max are the same value, normalization would be invalid: (val - min)/(max - min) -> 0/0
                # We thus set the min value to 0, so the normalized value is 1. The same works for denormalization:  val (<- 1) * (max - min) -> 1.0 * max
                new_scaler_values["min"][new_scaler_values["min"] == new_scaler_values["max"]] = 0.0
                for relevant_key in relevant_keys:
                    if not relevant_key in validated_scaler_values[key] or validated_scaler_values[key][relevant_key] is None:
                        validated_scaler_values[key][relevant_key] = new_scaler_values[relevant_key]

                # If there are any values that are still the same (min = max = 0) -> raise error as we cannot calculate the normalized and denormalized values in a safe way
                if any(validated_scaler_values[key]["max"][validated_scaler_values[key]["max"] == validated_scaler_values[key]["min"]] == 0.0):
                    raise RuntimeError(f"Could not calculate min max scaler values for {key}. All values are 0.0.")

        return validated_scaler_values

    def _pad_tensors(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.pad_start == 0 and self.pad_end == 0:
            return tensor

        padded_tensor = torch.zeros((tensor.shape[0] + self.pad_start + self.pad_end, *tensor.shape[1:]), dtype=torch.float32)
        padded_tensor[self.pad_start:self.pad_start + tensor.shape[0], ...] = tensor
        padded_tensor[:self.pad_start, ...] = tensor[0, ...]
        padded_tensor[self.pad_start + tensor.shape[0]:, ...] = tensor[-1, ...]
        return padded_tensor

    def _pad_list(self, list: List[Any]) -> List[Any]:
        if self.pad_start == 0 and self.pad_end == 0:
            return list

        padded_list = [list[0]] * self.pad_start + list + [list[-1]] * self.pad_end
        return padded_list

    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        """Get item from dataset.

        Args:
            index (int): Index of the trajectory

        Returns:
            Dict[str, Union[torch.Tensor]]: Dictionary containing the trajectory.
        """

        # Get the trajectory data
        trajectory = {key: val for key, val in self.trajectories[index].items() if not key in self.image_keys}

        # Process the image data
        for key in self.image_keys:
            # Check if the image is already a tensor or has to be loaded from file
            if isinstance(image_data := self.trajectories[index][key], torch.Tensor):
                # Directly use the tensor
                image_tensor = self.image_transform[key](image_data)
            else:
                # Preallocate the tensor
                image_tensor = torch.zeros((self.trajectory_lengths[index],) + self.image_shape[key], dtype=torch.float32)
                # Load the images from files
                for image_index, image_path in enumerate(self.trajectories[index][key]):
                    image = read_image(str(image_path.absolute()))
                    image_tensor[image_index, ...] = self.image_transform[key](image)
                # Move the tensor to the device
                image_tensor = image_tensor.to(self.device)

            # Add the image tensor to the trajectory
            trajectory[key] = image_tensor

        return trajectory

    def __len__(self) -> int:
        return len(self.trajectories)

    def save_scaler_values(self, path: Path) -> None:
        """Save the normalizer values to a file.

        Args:
            path (Path): Path to the file.
        """
        with open(path, "wb") as f:
            pickle.dump(self.scaler_values, f)

    def get_minimum_trajectory_length(self) -> int:
        """Return the minimum trajectory length."""
        return min([trajectory["length"] for trajectory in self.trajectories])

    def get_maximum_trajectory_length(self) -> int:
        """Return the maximum trajectory length."""
        return max([trajectory["length"] for trajectory in self.trajectories])

    def __calculate_velocity(self, position_key: str, velocity_key: str) -> None:
        for trajectory in self.trajectories:
            trajectory[velocity_key] = torch.zeros_like(trajectory[position_key])
            trajectory[velocity_key][1:] = (trajectory[position_key][1:] - trajectory[position_key][:-1]) / self.dt

    def get_statistics(self) -> Dict[str, Union[int, float]]:
        shortest_trajectory_length = self.get_minimum_trajectory_length()
        longest_trajectory_length = self.get_maximum_trajectory_length()

        return {
            "shortest_trajectory_length_steps": shortest_trajectory_length,
            "shortest_trajectory_length_seconds": shortest_trajectory_length * self.dt,
            "longest_trajectory_length_steps": longest_trajectory_length,
            "longest_trajectory_length_seconds": longest_trajectory_length * self.dt,
            "number_of_trajectories": len(self.trajectories),
            "mean_length_steps": float(np.mean([trajectory["length"] for trajectory in self.trajectories])),
            "mean_length_seconds": float(np.mean([trajectory["length"] for trajectory in self.trajectories]) * self.dt),
            "median_length_steps": float(np.median([trajectory["length"] for trajectory in self.trajectories])),
            "median_length_seconds": float(np.median([trajectory["length"] for trajectory in self.trajectories]) * self.dt),
        }

    def get_trajectory_dir(self, index) -> Path:
        """Return the directory where the trajectory associated with the specified index (= TrajectoryDataset[index]) was loaded from."""
        return self.trajectory_dirs[index]


class SubsequenceTrajectoryDataset(TrajectoryDataset):
    """Dataset class for loading trajectories and splitting them into subsequences.

    Args:
        subsequence_length (int): Length of the subsequences.
        trajectory_dirs (List[Path]): List of paths to directories containing trajectories. Each directory should contain a single trajectory. The trajectory should be stored as a numpy file with the keys specified in keys.
        keys (List[str]): List of keys to load from the trajectory files.
        dt (float): Time step of the trajectories.
        target_dt (Optional[float]): Target time step of the trajectories. If None, the trajectories are not resampled. Defaults to None.
        normalize_keys (Union[bool, List[str]]): Whether to normalize the keys. If True, all keys are normalized. If a list of keys is provided, only the keys in the list are normalized. Defaults to False.
        normalize_symmetrically (bool): Whether to normalize to [-1, 1] (True) or [0, 1] (False). Defaults to False.
        standardize_keys (Union[bool, List[str]]): Whether to standardize the keys. If True, all keys are standardized. If a list of keys is provided, only the keys in the list are standardized. Defaults to False.
        scaler_values (Optional[Dict[str, Dict[str, torch.Tensor]]]): Dictionary of normalizer values. If None, the normalizer values are calculated from the data. Defaults to None.
        image_keys (Optional[List[str]]): List of keys to load images from. Defaults to None. These images can be loaded from a directory of png files or a single .npz file.
        image_sizes (Optional[List[Tuple[int, int]]]): List of sizes of the images to load. If None, the images are loaded in their original size. Defaults to None.
        crop_sizes (Optional[List[Tuple[int, int]]]): List of sizes of the images to crop to. If None, the images are not cropped. Defaults to None.
        random_crop (Union[List[bool], bool]): Whether to randomly crop the images or crop them from the center. If True, all images are randomly cropped. If a list of booleans is provided, only the images with a corresponding True value are randomly cropped. Defaults to True.
        normalize_images (Union[List[bool], bool]): Whether to normalize the images. If True, all images are normalized. If a list of booleans is provided, only the images with a corresponding True value are normalized. Defaults to True.
        calculate_velocities_from_to (List[Tuple[str, str]]): List of tuples of keys to calculate velocities from and to. Defaults to [].
        recalculate_velocities_from_to (List[Tuple[str, str]]): List of tuples of keys to recalculate velocities after normalization of position values. Defaults to [].
        ignore_shorter_trajectories (bool): Whether to ignore trajectories that are shorter than the subsequence length. Defaults to False.
        pad_start (int): Number of steps to pad at the start of each trajectory. Defaults to 0.
        pad_end (int): Number of steps to pad at the end of each trajectory. Defaults to 0.

    TODO:
        - Add support to set a stride for the subsequences. Currently the stride is always one.
        - Add support for non-overlapping subsequences.
    """

    def __init__(
        self,
        subsequence_length: int,
        trajectory_dirs: List[Path],
        keys: List[str],
        dt: float,
        target_dt: Optional[float] = None,
        normalize_keys: Union[bool, List[str]] = False,
        normalize_symmetrically: bool = False,
        standardize_keys: Union[bool, List[str]] = False,
        scaler_values: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
        image_keys: Optional[List[str]] = None,
        image_sizes: Optional[List[Tuple[int, int]]] = None,  # HxW
        crop_sizes: Optional[List[Tuple[int, int]]] = None,  # HxW
        random_crop: Union[List[bool], bool] = True,
        normalize_images: Union[List[bool], bool] = True,
        calculate_velocities_from_to: List[Tuple[str, str]] = [],
        recalculate_velocities_from_to: List[Tuple[str, str]] = [],
        ignore_shorter_trajectories: bool = False,
        pad_start: int = 0,
        pad_end: int = 0,
    ):
        super().__init__(
            trajectory_dirs=trajectory_dirs,
            keys=keys,
            dt=dt,
            target_dt=target_dt,
            normalize_keys=normalize_keys,
            normalize_symmetrically=normalize_symmetrically,
            standardize_keys=standardize_keys,
            scaler_values=scaler_values,
            image_keys=image_keys,
            image_sizes=image_sizes,
            crop_sizes=crop_sizes,
            random_crop=random_crop,
            normalize_images=normalize_images,
            calculate_velocities_from_to=calculate_velocities_from_to,
            recalculate_velocities_from_to=recalculate_velocities_from_to,
            pad_start=pad_start,
            pad_end=pad_end,
        )

        self.subsequence_length = subsequence_length
        self.ignore_shorter_trajectories = ignore_shorter_trajectories
        self.__create_subsequence_indices(subsequence_length)

    def split(self, split_ratios: List[float], shuffle: bool = True) -> Tuple[List["SubsequenceTrajectoryDataset"], List[np.ndarray]]:
        """Splits the dataset into multiple datasets.

        Args:
            split_ratios (List[float]): List of fractions to split the dataset into. The fractions must sum to one.
            shuffle (bool, optional): Whether to shuffle the dataset before splitting. Defaults to True.

        Raises:
            ValueError: If the fractions do not sum to one.
            ValueError: If any of the fractions are too small.

        Returns:
            Tuple[List[SubsequenceTrajectoryDataset], torch.Tensor]: List of datasets and a tensor of indices for each dataset.
        """

        # Ensure the fractions sum to one
        if np.sum(split_ratios) != 1.0:
            raise ValueError("Fractions must sum to one.")

        # Create an array of indices
        trajectory_indices = np.arange(len(self.trajectories))

        # Shuffle the indices if requested
        if shuffle:
            np.random.shuffle(trajectory_indices)

        # Convert fractions to index counts
        fractions_as_counts = [int(len(trajectory_indices) * fraction) for fraction in split_ratios]

        # Adjust last count to capture any rounding issues
        fractions_as_counts[-1] = len(trajectory_indices) - np.sum(fractions_as_counts[:-1])

        # Ensure that each count is at least one
        if any([count < 1 for count in fractions_as_counts]):
            raise ValueError("Fractions are too small.")

        # Split the indices based on counts
        split_indices = np.split(trajectory_indices, np.cumsum(fractions_as_counts[:-1]))

        # Create a dataset for each split by copying the current dataset and adjusting the trajectories and trajectory_lengths
        split_datasets = []
        for split in split_indices:
            dataset = deepcopy(self)
            dataset.trajectory_dirs = [dataset.trajectory_dirs[index] for index in split]
            dataset.trajectories = [dataset.trajectories[index] for index in split]
            dataset.trajectory_lengths = [dataset.trajectory_lengths[index] for index in split]
            # Recalculate the subsequence indices for each dataset
            dataset.__create_subsequence_indices(self.subsequence_length)
            split_datasets.append(dataset)

        return split_datasets, split_indices

    def __create_subsequence_indices(self, subsequence_length: int) -> None:
        subsequence_indices = []
        invalid_indices = []
        for trajectory_index in range(len(self.trajectories)):
            trajectory_length = self.trajectory_lengths[trajectory_index]

            if trajectory_length < subsequence_length:
                invalid_indices.append(trajectory_index)
                continue

            start_indices = torch.arange(0, trajectory_length - subsequence_length + 1, dtype=torch.int64)
            end_indices = start_indices + subsequence_length
            trajectory_indices = torch.ones_like(start_indices) * trajectory_index

            subsequence_indices.append(torch.stack([trajectory_indices, start_indices, end_indices], dim=1))

        if len(invalid_indices) > 0:
            if self.ignore_shorter_trajectories:
                print(f"WARNING: Some trajectories are shorter than the requested subsequence length of {subsequence_length}. These trajectories will be ignored.")
            else:
                import tabulate

                header = ["Path", "Length"]
                table = [[self.trajectory_dirs[index], self.trajectory_lengths[index]] for index in invalid_indices]
                print(tabulate.tabulate(table, header, tablefmt="fancy_grid"))
                raise ValueError(f"Some trajectories are shorter than the requested subsequence length of {subsequence_length}. You can ignore these trajectories by using the `ignore_shorter_trajectories` argument.")

        # This is a tensor of shape (number_of_subsequences, 3)
        # where the first column is the trajectory index, the second column is the start index and the third column is the end index.
        # This way we can get the subsequence by indexing the trajectory and the start and end index.
        self.subsequence_indices = torch.cat(subsequence_indices, dim=0)

    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        """Get item from dataset.

        Args:
            index (int): Index of the subsequence. The subsequence is defined by the trajectory index, the start index and the end index.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing the subsequence.
        """

        # Get indices for the subsequence
        trajectory_index, start_index, end_index = self.subsequence_indices[index]

        # Get the subsequence for each key
        subsequence = {key: val[start_index:end_index] for key, val in self.trajectories[trajectory_index].items()}

        # Process the image keys
        for key in self.image_keys:
            # Check if the image is already a tensor or has to be loaded from file
            if isinstance(image_data := subsequence[key], torch.Tensor):
                # Directly use the tensor
                image_tensor = self.image_transform[key](image_data)
            else:
                # Preallocate the image tensor
                image_tensor = torch.zeros((self.subsequence_length,) + self.image_shape[key], dtype=torch.float32)
                # Load the images from files
                for image_index, image_path in enumerate(subsequence[key]):
                    image = read_image(str(image_path.absolute()))
                    image_tensor[image_index, ...] = self.image_transform[key](image)
                # Move the tensor to the device
                image_tensor = image_tensor.to(self.device)

            subsequence[key] = image_tensor

        return subsequence

    def __len__(self) -> int:
        return self.subsequence_indices.shape[0]

    def get_trajectory_dir(self, index) -> Path:
        """Return the directory where the (subsequence) trajectory associated with the specified index (= SubsequenceTrajectoryDataset[index]) was loaded from."""
        trajcetory_index, _, _ = self.subsequence_indices[index]
        return self.trajectory_dirs[trajcetory_index]
