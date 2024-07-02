import cv2
import numpy as np
from collections import deque, defaultdict
from functools import partial
from omegaconf import DictConfig, OmegaConf
from copy import deepcopy
from typing import Dict, List, Tuple, Optional, Union

from movement_primitive_diffusion.datasets.scalers import denormalize, destandardize, normalize, standardize
from movement_primitive_diffusion.utils.helper import deque_to_array

from sofa_env.scenes.deflect_spheres.deflect_spheres_env import ActionType, DeflectSpheresEnv, ObservationType, RenderMode, State


class DeflectSpheresEnvCompatLayer(DeflectSpheresEnv):
    def __init__(
        self,
        scaler_config: DictConfig,
        t_obs: int,
        rgb_image_size: Union[Tuple[int, int], List[Tuple[int, int]]],
        rgb_image_crop_size: Union[Tuple[int, int], List[Tuple[int, int]]],
        time_limit: Optional[int] = None,
        seed: Optional[int] = None,
        with_rgb_images: bool = False,
        *args,
        **kwargs,
    ):
        # Parse action and observation types
        kwargs["action_type"] = ActionType[kwargs["action_type"]]
        kwargs["observation_type"] = ObservationType[kwargs["observation_type"]]
        kwargs["render_mode"] = RenderMode[kwargs["render_mode"]]

        super().__init__(*args, **kwargs)

        if not self.single_agent:
            raise NotImplementedError("Only single agent is supported for now.")

        self.previous_pos = None
        self.time_step_counter = 0
        self.rgb_image_size = rgb_image_size if isinstance(rgb_image_size[0], int) else rgb_image_size[0]
        self.rgb_image_crop_size = rgb_image_crop_size if isinstance(rgb_image_crop_size[0], int) else rgb_image_crop_size[0]
        self.t_obs = t_obs
        self.time_limit = time_limit
        self.with_rgb_images = with_rgb_images

        self.observation_buffer = defaultdict(partial(deque, maxlen=self.t_obs))

        scaler_config = OmegaConf.to_container(scaler_config, resolve=True)
        self.normalize_keys: List[str] = scaler_config["normalize_keys"] if isinstance(scaler_config["normalize_keys"], list) else []
        self.standardize_keys: List[str] = scaler_config["standardize_keys"] if isinstance(scaler_config["standardize_keys"], list) else []
        self.normalize_symmetrically = scaler_config["normalize_symmetrically"]

        self.scaler_values = scaler_config["scaler_values"]

        # TODO: currently unused
        self.seed = seed

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
        cauter_state = self.right_cauter.get_state().astype(np.float32)
        if "right_ptsd_state" in self.normalize_keys:
            cauter_state = normalize(cauter_state, self.scaler_values["right_ptsd_state"], self.normalize_symmetrically)
        elif "right_ptsd_state" in self.standardize_keys:
            cauter_state = standardize(cauter_state, self.scaler_values["right_ptsd_state"])

        if not self.single_agent:
            left_cauter_state = self.left_cauter.get_state().astype(np.float32)
            if "left_ptsd_state" in self.normalize_keys:
                left_cauter_state = normalize(left_cauter_state, self.scaler_values["left_ptsd_state"], self.normalize_symmetrically)
            elif "left_ptsd_state" in self.standardize_keys:
                left_cauter_state = standardize(left_cauter_state, self.scaler_values["left_ptsd_state"])

            cauter_state = np.cat([cauter_state, left_cauter_state], axis=-1)

        return cauter_state

    @property
    def current_vel(self) -> np.ndarray:
        if self.time_step_counter == 0:
            vel = np.zeros(self.current_pos.shape, dtype=np.float32)
        else:
            vel = (self.current_pos - self.previous_pos) / self.dt

        return vel

    def get_sphere_positions(self) -> np.ndarray:
        sphere_positions = np.asarray([post.get_sphere_position() for post in self.posts]).ravel().astype(np.float32)
        if "sphere_positions" in self.normalize_keys:
            sphere_positions = normalize(sphere_positions, self.scaler_values["sphere_positions"], symmetric=self.normalize_symmetrically)
        elif "sphere_positions" in self.standardize_keys:
            sphere_positions = standardize(sphere_positions, self.scaler_values["sphere_positions"])
        return sphere_positions

    def get_active_sphere_position(self) -> np.ndarray:
        active_sphere_position = self.posts[self.active_post_index].get_sphere_position().astype(np.float32)
        if "active_sphere_position" in self.normalize_keys:
            active_sphere_position = normalize(active_sphere_position, self.scaler_values["active_sphere_position"], symmetric=self.normalize_symmetrically)
        elif "active_sphere_position" in self.standardize_keys:
            active_sphere_position = standardize(active_sphere_position, self.scaler_values["active_sphere_position"])
        return active_sphere_position

    def get_active_agent(self) -> np.ndarray:
        if self.single_agent:
            return np.array([], dtype=np.float32)
        else:
            return np.array(1 if self.posts[self.active_post_index].state == State.ACTIVE_LEFT else 0, dtype=np.float32)[None]  # 1 -> [1]

    def get_current_distance_tool_active_sphere(self) -> float:
        if self.posts[self.active_post_index].state == State.ACTIVE_LEFT:
            position_tool = self.left_cauter.get_cutting_center_position()
        else:
            position_tool = self.right_cauter.get_cutting_center_position()
        position_active_sphere = self.posts[self.active_post_index].get_sphere_position()
        distance = np.linalg.norm(position_tool - position_active_sphere)
        return distance

    def update_observation_buffer(self) -> None:
        self.observation_buffer["right_ptsd_state"].append(self.current_pos[:4])
        self.observation_buffer["right_ptsd_velocity"].append(self.current_vel[:4])
        # At this time point, the action has already been applied -> we can just use the current state
        self.observation_buffer["right_ptsd_state_action"].append(self.current_pos[:4])
        self.observation_buffer["right_ptsd_velocity_action"].append(self.current_vel[:4])

        right_pose = self.right_cauter.get_pose().astype(np.float32)
        if "right_pose" in self.normalize_keys:
            right_pose = normalize(right_pose, self.scaler_values["right_pose"], symmetric=self.normalize_symmetrically)
        elif "right_pose" in self.standardize_keys:
            right_pose = standardize(right_pose, self.scaler_values["right_pose"])
        self.observation_buffer["right_pose"].append(right_pose)

        if not self.single_agent:
            self.observation_buffer["left_ptsd_state"].append(self.current_pos[4:])
            self.observation_buffer["left_ptsd_velocity"].append(self.current_vel[4:])
            # At this time point, the action has already been applied -> we can just use the current state
            self.observation_buffer["left_ptsd_state_action"].append(self.current_pos[4:])
            self.observation_buffer["left_ptsd_velocity_action"].append(self.current_vel[4:])

            left_pose = self.left_cauter.get_pose().astype(np.float32)
            if "left_pose" in self.normalize_keys:
                left_pose = normalize(left_pose, self.scaler_values["left_pose"], symmetric=self.normalize_symmetrically)
            elif "left_pose" in self.standardize_keys:
                left_pose = standardize(left_pose, self.scaler_values["left_pose"])
            self.observation_buffer["left_pose"].append(left_pose)

            self.observation_buffer["active_agent"].append(self.get_active_agent())

        self.observation_buffer["sphere_positions"].append(self.get_sphere_positions())
        self.observation_buffer["active_sphere_position"].append(self.get_active_sphere_position())
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
        return deque_to_array(self.observation_buffer)

    def step(self, action: np.ndarray):
        self.time_step_counter += 1
        self.previous_pos = deepcopy(self.current_pos)

        if self.action_type == ActionType.CONTINUOUS:
            key_suffix = None  # actions already scaled to [-1, 1]
        elif self.action_type == ActionType.POSITION:
            key_suffix = "_state"
        elif self.action_type == ActionType.VELOCITY:
            key_suffix = "_velocity"
        else:
            raise NotImplementedError

        if not self.single_agent:
            if (key := f"left_ptsd{key_suffix}") in self.normalize_keys:
                action[4:] = denormalize(action[4:], self.scaler_values[key], symmetric=self.normalize_symmetrically)
            elif (key := f"left_ptsd_{key_suffix}") in self.standardize_keys:
                action[4:] = destandardize(action[4:], self.scaler_values[key])

        if (key := f"right_ptsd{key_suffix}") in self.normalize_keys:
            action[:4] = denormalize(action[:4], self.scaler_values[key], symmetric=self.normalize_symmetrically)
        elif (key := f"right_ptsd_{key_suffix}") in self.standardize_keys:
            action[:4] = destandardize(action[:4], self.scaler_values[key])

        # Clip actions to valid range
        action = np.clip(action, self.action_space_limits["low"], self.action_space_limits["high"])

        # flip order of positions in action from right, left to left, right
        indices = np.concatenate([np.arange(4, len(action)), np.arange(0, 4)])
        observation, reward, terminated, truncated, info = super().step(action[indices])
        self.update_observation_buffer()
        if self.time_limit is not None:
            if self.time_step_counter >= self.time_limit:
                truncated = True
        return observation, reward, terminated, truncated, info

    def reset(self, sphere_position: Optional[np.ndarray] = None, cauter_state: Optional[np.ndarray] = None, *args, **kwargs):
        self.time_step_counter = 0
        reset_results = super().reset(*args, **kwargs)
        animate_a_few = False

        if sphere_position is not None:
            self.set_active_sphere_position(sphere_position)
            animate_a_few = True

        if cauter_state is not None:
            self.set_cauter_state(cauter_state)
            animate_a_few = True

        if animate_a_few:
            # Animate several timesteps without actions until simulation settles
            for _ in range(10):
                self.sofa_simulation.animate(self._sofa_root_node, self.time_step)
                self._update_rgb_buffer()

        # Fill observation buffer with initial values
        for _ in range(self.t_obs):
            self.update_observation_buffer()
            if not self.single_agent:
                self.observation_buffer["left_ptsd_state_action"].append(self.observation_buffer["left_ptsd_state"][-1])
                self.observation_buffer["left_ptsd_velocity_action"].append(np.zeros_like(self.observation_buffer["left_ptsd_state"][-1]))
            self.observation_buffer["right_ptsd_state_action"].append(self.observation_buffer["right_ptsd_state"][-1])
            self.observation_buffer["right_ptsd_velocity_action"].append(np.zeros_like(self.observation_buffer["right_ptsd_state"][-1]))

        return reset_results

    def set_active_sphere_position(self, position: np.ndarray):
        self.posts[self.active_post_index].set_position(position[:2], position[-1])

    def set_cauter_state(self, state: np.ndarray):
        self.right_cauter.set_state(state)
