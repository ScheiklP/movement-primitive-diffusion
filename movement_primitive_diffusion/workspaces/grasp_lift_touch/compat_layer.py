import cv2
import numpy as np

from collections import deque, defaultdict
from functools import partial
from omegaconf import DictConfig, OmegaConf
from copy import deepcopy
from typing import Dict, List, Tuple, Optional, Union

from movement_primitive_diffusion.datasets.scalers import denormalize, destandardize, normalize, standardize
from movement_primitive_diffusion.utils.helper import deque_to_array

from sofa_env.scenes.grasp_lift_touch.grasp_lift_touch_env import ActionType, GraspLiftTouchEnv, ObservationType, Phase, RenderMode


class GraspLiftTouchEnvCompatLayer(GraspLiftTouchEnv):
    def __init__(
        self,
        scaler_config: DictConfig,
        extra_scaler_values: DictConfig,
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

        reward_mount_dict = OmegaConf.to_container(kwargs["reward_amount_dict"], resolve=True, throw_on_missing=True)
        kwargs["reward_amount_dict"] = {Phase[key]: value for key, value in reward_mount_dict.items()}

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

        self.extra_scaler_values = OmegaConf.to_container(extra_scaler_values, resolve=True)
        for key in extra_scaler_values:
            for metric in ["min", "max"]:
                self.extra_scaler_values[key][metric] = np.array(self.extra_scaler_values[key][metric], dtype=np.float32)
        assert all([key in self.extra_scaler_values for key in ["gallbladder_state"]])

    @property
    def dt(self) -> float:
        """Control frequency of the environment"""
        return self.time_step * self.frame_skip

    @property
    def current_pos(self) -> np.ndarray:
        gripper_state = self.gripper.articulated_state.astype(np.float32)
        cauter_state = self.cauter.ptsd_state.astype(np.float32)

        if "gripper_tpsda_state" in self.normalize_keys:
            gripper_state = normalize(gripper_state, self.scaler_values["gripper_tpsda_state"], self.normalize_symmetrically)
        elif "gripper_tpsda_state" in self.standardize_keys:
            gripper_state = standardize(gripper_state, self.scaler_values["gripper_tpsda_state"])

        if "cauter_tpsd_state" in self.normalize_keys:
            cauter_state = normalize(cauter_state, self.scaler_values["cauter_tpsd_state"], self.normalize_symmetrically)
        elif "cauter_tpsd_state" in self.standardize_keys:
            cauter_state = standardize(cauter_state, self.scaler_values["cauter_tpsd_state"])

        tool_state = np.concatenate([gripper_state, cauter_state], axis=-1)

        return tool_state

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
        current_pos = self.current_pos
        current_vel = self.current_vel
        self.observation_buffer["gripper_tpsda_state"].append(current_pos[:5])
        self.observation_buffer["gripper_tpsda_velocity"].append(current_vel[:5])
        self.observation_buffer["gripper_tpsda_state_action"].append(current_pos[:5])
        self.observation_buffer["gripper_tpsda_velocity_action"].append(current_vel[:5])
        self.observation_buffer["cauter_tpsd_state"].append(current_pos[5:])
        self.observation_buffer["cauter_tpsd_velocity"].append(current_vel[5:])
        self.observation_buffer["cauter_tpsd_state_action"].append(current_pos[5:])
        self.observation_buffer["cauter_tpsd_velocity_action"].append(current_vel[5:])

        self.observation_buffer["gripper_pose"].append(self.gripper.get_pose().astype(np.float32))
        self.observation_buffer["cauter_pose"].append(self.cauter.get_pose().astype(np.float32))
        # Gallbladder state is a vector of 27 values. Normalizing that with 27 values would be a bit weird,
        # because they are actually 9 3D points -> we actually just need 3 values to normalize the position.
        gallbladder_state = self.gallbladder_surface_mechanical_object.position.array()[self.gallbladder_tracking_indices].astype(np.float32)
        self.observation_buffer["gallbladder_state"].append(gallbladder_state.ravel())
        self.observation_buffer["gallbladder_state_normalized"].append(
            normalize(
                gallbladder_state,
                self.extra_scaler_values["gallbladder_state"],
                self.normalize_symmetrically,
            ).ravel()
        )

        self.observation_buffer["phase"].append(np.array([self.active_phase.value], dtype=np.float32))

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
        for key in ["gripper_pose", "cauter_pose", "gallbladder_state", "phase", "time"]:
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

        if (key := f"gripper_tpsda{key_suffix}") in self.normalize_keys:
            action[:5] = denormalize(action[:5], self.scaler_values[key], symmetric=self.normalize_symmetrically)
        elif (key := f"gripper_tpsda{key_suffix}") in self.standardize_keys:
            action[:5] = destandardize(action[:5], self.scaler_values[key])
        if (key := f"cauter_tpsd{key_suffix}") in self.normalize_keys:
            action[5:] = denormalize(action[5:], self.scaler_values[key], symmetric=self.normalize_symmetrically)
        elif (key := f"cauter_tpsd{key_suffix}") in self.standardize_keys:
            action[5:] = destandardize(action[5:], self.scaler_values[key])

        # Clip actions to valid range
        # Cut off the last value from the action_space_limits, as we do not use the cauter activation
        action = np.clip(action, self.action_space.low[:-1], self.action_space.high[:-1])

        # Execute the action
        observation, reward, terminated, truncated, info = super().step(action)

        if self.calculate_accelerations:
            tissue_velocity = self.gallbladder.get_tissue_velocities()
            tissue_acceleration = (tissue_velocity - self.previous_tissue_velocity) / self.dt
            self.previous_tissue_velocity[:] = tissue_velocity
            info["tissue_acceleration"] = np.linalg.norm(tissue_acceleration, axis=1).mean()
            info["tool_acceleration"] = self.current_acc

        info["tool_positions"] = np.concatenate([self.gripper.get_pose()[:3].astype(np.float32), self.cauter.get_pose()[:3].astype(np.float32)])

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
            self.previous_tissue_velocity = self.gallbladder.get_tissue_velocities().copy()

        return reset_results


if __name__ == "__main__":
    import itertools

    env = GraspLiftTouchEnvCompatLayer(
        scaler_config=DictConfig(
            {
                "normalize_keys": ["gripper_tpsda_state", "cauter_tpsd_state", "time"],
                "standardize_keys": None,
                "normalize_symmetrically": True,
                "scaler_values": {
                    "gripper_tpsda_state": {
                        "mean": None,
                        "std": None,
                        "min": [-75.0, -40.0, -100.0, 12.0, 0.0],
                        "max": [75.0, 75.0, 100.0, 300.0, 60.0],
                    },
                    "cauter_tpsd_state": {
                        "mean": None,
                        "std": None,
                        "min": [-75.0, -40.0, -100.0, 12.0],
                        "max": [75.0, 75.0, 100.0, 300.0],
                    },
                    "time": {
                        "mean": None,
                        "std": None,
                        "min": [0.0],
                        "max": [1000.0 * 0.1],
                    },
                },
            }
        ),
        extra_scaler_values=DictConfig(
            {
                "gallbladder_state": {
                    "mean": None,
                    "std": None,
                    "min": [-0.1, -0.1, -0.1],
                    "max": [0.1, 0.1, 0.1],
                },
            }
        ),
        t_obs=3,
        rgb_image_size=(84, 84),
        rgb_image_crop_size=(76, 76),
        time_limit=1000,
        action_type="POSITION",
        observation_type="STATE",
        render_mode="HUMAN",
    )

    env.reset()
    seed_list = itertools.cycle([42, 1337, 0, 1, 99])

    counter = 0
    while True:
        env.step(env.current_pos)
        observation_dict = env.get_observation_dict()
        env.render()
        counter += 1
        if counter % 100 == 0:
            env.reset(seed=next(seed_list))
