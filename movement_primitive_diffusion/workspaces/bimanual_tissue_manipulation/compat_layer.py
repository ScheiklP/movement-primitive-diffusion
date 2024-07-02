import cv2
import numpy as np

from collections import deque, defaultdict
from functools import partial
from omegaconf import DictConfig, OmegaConf
from copy import deepcopy
from typing import Dict, List, Tuple, Optional, Union

from movement_primitive_diffusion.datasets.scalers import denormalize, destandardize, normalize, standardize
from movement_primitive_diffusion.utils.helper import deque_to_array

from sofa_env.scenes.bimanual_tissue_manipulation.bimanual_tissue_manipulation_env import ActionType, BimanualTissueManipulationEnv, ObservationType, RenderMode


class BimanualTissueManipulationEnvCompatLayer(BimanualTissueManipulationEnv):
    def __init__(
        self,
        scaler_config: DictConfig,
        t_obs: int,
        rgb_image_size: Union[Tuple[int, int], List[Tuple[int, int]]],
        rgb_image_crop_size: Union[Tuple[int, int], List[Tuple[int, int]]],
        time_limit: Optional[int] = None,
        seed: Optional[int] = None,  # NOTE: Unused
        with_rgb_images: bool = False,
        calculate_accelerations: bool = True,
        *args,
        **kwargs,
    ):
        # Parse action and observation types
        kwargs["action_type"] = ActionType[kwargs["action_type"]]
        kwargs["observation_type"] = ObservationType[kwargs["observation_type"]]
        kwargs["render_mode"] = RenderMode[kwargs["render_mode"]]
        kwargs["reward_amount_dict"] = OmegaConf.to_container(kwargs["reward_amount_dict"], resolve=True, throw_on_missing=True)

        super().__init__(*args, **kwargs)

        self.previous_pos = None
        self.time_step_counter = 0
        self.rgb_image_size = rgb_image_size if isinstance(rgb_image_size[0], int) else rgb_image_size[0]
        self.rgb_image_crop_size = rgb_image_crop_size if isinstance(rgb_image_crop_size[0], int) else rgb_image_crop_size[0]
        self.t_obs = t_obs
        self.time_limit = time_limit
        self.with_rgb_images = with_rgb_images
        self.calculate_accelerations = calculate_accelerations
        self.previous_tissue_velocity: np.ndarray

        self.observation_buffer = defaultdict(partial(deque, maxlen=self.t_obs))

        scaler_config = OmegaConf.to_container(scaler_config, resolve=True)
        self.normalize_keys: List[str] = scaler_config["normalize_keys"] if isinstance(scaler_config["normalize_keys"], list) else []
        self.standardize_keys: List[str] = scaler_config["standardize_keys"] if isinstance(scaler_config["standardize_keys"], list) else []
        self.normalize_symmetrically = scaler_config["normalize_symmetrically"]

        self.scaler_values = scaler_config["scaler_values"]

        for key in self.normalize_keys:
            assert key in self.scaler_values, f"Key {key} not found in scaler values."
            for metric in ["min", "max"]:
                assert metric in self.scaler_values[key] and self.scaler_values[key][metric] is not None, f"Key {key} does not have {metric} in scaler values."
                self.scaler_values[key][metric] = np.array(self.scaler_values[key][metric], dtype=np.float32)

        for key in self.standardize_keys:
            assert key in self.scaler_values, f"Key {key} not found in scaler values."
            for metric in ["mean", "std"]:
                assert metric in self.scaler_values[key] and self.scaler_values[key][metric] is not None, f"Key {key} does not have {metric} in scaler values."
                self.scaler_values[key][metric] = np.array(self.scaler_values[key][metric], dtype=np.float32)

    @property
    def dt(self) -> float:
        """Control frequency of the environment"""
        return self.time_step * self.frame_skip

    @property
    def current_pos(self) -> np.ndarray:
        left_gripper_state = self.left_gripper.get_state()[[0, 3]].astype(np.float32)
        if "left_pd_state" in self.normalize_keys:
            left_gripper_state = normalize(left_gripper_state, self.scaler_values["left_pd_state"], self.normalize_symmetrically)
        elif "left_pd_state" in self.standardize_keys:
            left_gripper_state = standardize(left_gripper_state, self.scaler_values["left_pd_state"])

        right_gripper_state = self.right_gripper.get_state()[[0, 3]].astype(np.float32)
        if "right_pd_state" in self.normalize_keys:
            right_gripper_state = normalize(right_gripper_state, self.scaler_values["right_pd_state"], self.normalize_symmetrically)
        elif "right_pd_state" in self.standardize_keys:
            right_gripper_state = standardize(right_gripper_state, self.scaler_values["right_pd_state"])

        gripper_state = np.concatenate([left_gripper_state, right_gripper_state], axis=-1)

        return gripper_state

    @property
    def current_vel(self) -> np.ndarray:
        if self.time_step_counter == 0:
            vel = np.zeros(self.current_pos.shape, dtype=np.float32)
        else:
            vel = (self.current_pos - self.previous_pos) / self.dt

        return vel

    @property
    def current_acc(self) -> np.ndarray:
        if self.time_step_counter == 0:
            acc = np.zeros(self.current_pos.shape, dtype=np.float32)
        else:
            acc = (self.current_vel - self.previous_vel) / self.dt

        return acc

    def update_observation_buffer(self) -> None:
        self.observation_buffer["left_pd_state"].append(self.current_pos[:2])
        self.observation_buffer["left_pd_velocity"].append(self.current_vel[:2])
        # At this time point, the action has already been applied -> we can just use the current state
        self.observation_buffer["left_pd_state_action"].append(self.current_pos[:2])
        self.observation_buffer["left_pd_velocity_action"].append(self.current_vel[:2])
        self.observation_buffer["left_pose"].append(self.left_gripper.get_pose().astype(np.float32))
        self.observation_buffer["right_pd_state"].append(self.current_pos[2:])
        self.observation_buffer["right_pd_velocity"].append(self.current_vel[2:])
        # At this time point, the action has already been applied -> we can just use the current state
        self.observation_buffer["right_pd_state_action"].append(self.current_pos[2:])
        self.observation_buffer["right_pd_velocity_action"].append(self.current_vel[2:])
        self.observation_buffer["right_pose"].append(self.right_gripper.get_pose().astype(np.float32))
        self.observation_buffer["marker_positions"].append(self.tissue.get_marker_positions()[:, :2].astype(np.float32).ravel())
        self.observation_buffer["marker_positions_in_image"].append(self.get_marker_positions_in_image().astype(np.float32).ravel())
        self.observation_buffer["target_positions"].append(self.target_positions[:, :2].astype(np.float32).ravel())
        self.observation_buffer["target_positions_in_image"].append(self.get_target_positions_in_image().astype(np.float32).ravel())
        self.observation_buffer["time"].append(np.array([self.time_step_counter * self.dt], dtype=np.float32))

        if self.with_rgb_images:
            original_image = self._update_rgb_buffer()
            resized_image = cv2.resize(original_image, self.rgb_image_size)
            cropped_image = resized_image[
                self.rgb_image_size[0] // 2 - self.rgb_image_crop_size[0] // 2 : self.rgb_image_size[0] // 2 + self.rgb_image_crop_size[0] // 2,
                self.rgb_image_size[1] // 2 - self.rgb_image_crop_size[1] // 2 : self.rgb_image_size[1] // 2 + self.rgb_image_crop_size[1] // 2,
            ]
            self.observation_buffer["rgb"].append(np.moveaxis(cropped_image, -1, 0).astype(np.float32) / 255.0)

    def get_observation_dict(self) -> Dict[str, np.ndarray]:
        observation_dict = deque_to_array(self.observation_buffer)

        # Scale the not yet scaled values
        for key in ["left_pose", "right_pose", "marker_positions", "target_positions", "time"]:
            if key in self.normalize_keys:
                observation_dict[key] = normalize(observation_dict[key], self.scaler_values[key], self.normalize_symmetrically)
            elif key in self.standardize_keys:
                observation_dict[key] = standardize(observation_dict[key], self.scaler_values[key])

        return observation_dict

    def step(self, action: np.ndarray):
        self.time_step_counter += 1
        self.previous_pos = deepcopy(self.current_pos)
        self.previous_vel = deepcopy(self.current_vel)

        if self.action_type == ActionType.CONTINUOUS:
            key_suffix = None  # actions already scaled to [-1, 1]
        elif self.action_type == ActionType.POSITION:
            key_suffix = "_state"
        elif self.action_type == ActionType.VELOCITY:
            key_suffix = "_velocity"
        else:
            raise NotImplementedError

        if (key := f"left_pd{key_suffix}") in self.normalize_keys:
            action[:2] = denormalize(action[:2], self.scaler_values[key], symmetric=self.normalize_symmetrically)
        elif (key := f"left_pd_{key_suffix}") in self.standardize_keys:
            action[:2] = destandardize(action[:2], self.scaler_values[key])
        if (key := f"right_pd{key_suffix}") in self.normalize_keys:
            action[2:] = denormalize(action[2:], self.scaler_values[key], symmetric=self.normalize_symmetrically)
        elif (key := f"right_pd_{key_suffix}") in self.standardize_keys:
            action[2:] = destandardize(action[2:], self.scaler_values[key])

        # Clip actions to valid range
        action = np.clip(action, self.action_space_limits["low"], self.action_space_limits["high"])

        # Execute the action
        observation, reward, terminated, truncated, info = super().step(action)

        if self.calculate_accelerations:
            tissue_velocity = self.tissue.get_tissue_velocities()
            tissue_acceleration = (tissue_velocity - self.previous_tissue_velocity) / self.dt
            self.previous_tissue_velocity[:] = tissue_velocity
            info["tissue_acceleration"] = np.linalg.norm(tissue_acceleration, axis=1).mean()
            info["tool_acceleration"] = self.current_acc

        info["tool_positions"] = np.concatenate([self.left_gripper.get_pose()[:3].astype(np.float32), self.right_gripper.get_pose()[:3].astype(np.float32)])

        # Update the observation buffer
        self.update_observation_buffer()

        # Check if the time limit has been reached
        if self.time_limit is not None:
            if self.time_step_counter >= self.time_limit:
                truncated = True

        return observation, reward, terminated, truncated, info

    def reset(self, *args, **kwargs):
        self.time_step_counter = 0
        reset_results = super().reset(*args, **kwargs)

        # Fill observation buffer with initial values
        for _ in range(self.t_obs):
            self.update_observation_buffer()

        if self.calculate_accelerations:
            self.previous_tissue_velocity = self.tissue.get_tissue_velocities().copy()

        return reset_results

    def get_distances_to_targets(self) -> Dict:
        marker_positions = self.tissue.get_marker_positions()[:, :2].astype(np.float32)
        marker_positions_in_image = self.get_marker_positions_in_image().astype(np.float32)
        target_positions = self.target_positions[:, :2].astype(np.float32)
        target_positions_in_image = self.get_target_positions_in_image().astype(np.float32)

        distances = np.linalg.norm(marker_positions - target_positions, axis=-1)
        distances_in_image = np.linalg.norm(marker_positions_in_image - target_positions_in_image, axis=-1)

        return {
            "distances": distances,
            "distances_in_image": distances_in_image,
        }
