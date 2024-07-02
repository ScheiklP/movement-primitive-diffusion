from typing import Optional, Union
from functools import partial
from copy import deepcopy
from collections import deque, defaultdict
import warnings

import numpy as np
import cv2
from omegaconf import DictConfig, OmegaConf

import gymnasium as gym

from sofa_env.scenes.rope_threading.rope_threading_env import RopeThreadingEnv, ActionType, ObservationType, RenderMode
from sofa_env.scenes.rope_threading.sofa_objects.eye import Eye

from movement_primitive_diffusion.datasets.scalers import denormalize, destandardize, normalize, standardize
from movement_primitive_diffusion.utils.helper import deque_to_array


class RopeThreadingEnvCompatLayer(RopeThreadingEnv):
    def __init__(
        self,
        scaler_config: DictConfig,
        extra_scaler_values: DictConfig,
        t_obs: int,
        rgb_image_size: Union[tuple[int, int], list[tuple[int, int]]],
        rgb_image_crop_size: Union[tuple[int, int], list[tuple[int, int]]],
        sim_dt: float = 1 / 100,
        control_dt: float = 1 / 10,
        time_limit: Optional[int] = None,
        seed: Optional[int] = None,  # NOTE: Unused
        with_rgb_images: bool = False,
        calculate_accelerations: bool = True,
        *args,
        **kwargs,
    ):
        assert 0 < sim_dt <= control_dt, "Simulation dt must be less than or equal to a positive control dt"
        ### SCENE
        # default values
        kwargs.setdefault("frame_skip", int(control_dt // sim_dt))
        kwargs.setdefault("time_step", sim_dt)
        kwargs.setdefault("settle_steps", 10)
        kwargs.setdefault("image_shape", (256, 256))
        kwargs.setdefault("fraction_of_rope_to_pass", 0.05)
        # parse action and observation types
        if isinstance((at := kwargs.setdefault("action_type", ActionType.POSITION)), str):
            kwargs["action_type"] = ActionType[at]
        if isinstance((ot := kwargs.setdefault("observation_type", ObservationType.STATE)), str):
            kwargs["observation_type"] = ObservationType[ot]
        if isinstance((rm := kwargs.setdefault("render_mode", RenderMode.HEADLESS)), str):
            kwargs["render_mode"] = RenderMode[rm]
        # rewards
        kwargs["reward_amount_dict"] = OmegaConf.to_container(kwargs["reward_amount_dict"], resolve=True, throw_on_missing=True)
        # env modicifactions: single agent, single eyelet
        kwargs.setdefault("only_right_gripper", True)
        kwargs.setdefault("individual_agents", True)

        create_scene_kwargs = kwargs.setdefault("create_scene_kwargs", {})
        if isinstance(create_scene_kwargs, DictConfig):
            kwargs["create_scene_kwargs"] = OmegaConf.to_container(create_scene_kwargs, resolve=True)
            create_scene_kwargs = kwargs["create_scene_kwargs"]

        create_scene_kwargs.setdefault("eye_config", [(60, 10, 0, 90)])
        create_scene_kwargs.setdefault(
            "eye_reset_noise",
            {
                "low": np.array([-20.0, -20.0, 0.0, -35]),
                "high": np.array([20.0, 20.0, 0.0, 35]),
            },
        )
        create_scene_kwargs.setdefault("randomize_gripper", True)
        create_scene_kwargs.setdefault("randomize_grasp_index", False)
        create_scene_kwargs.setdefault("start_grasped", True)
        create_scene_kwargs.setdefault("gripper_and_rope_same_collision_group", True)

        super().__init__(*args, **kwargs)
        if not np.allclose(self.time_step * self.frame_skip, control_dt):
            warnings.warn("dt of environment does not match specified control dt. Did you override frame_skip?")

        # keep gripper jaws closed
        # i.e. strip the 'a' from the 'tpsda' action space
        assert self.action_space.shape == (5,), "Expected action space to be 5-dimensional."
        self.true_action_space = deepcopy(self.action_space)
        action_limits_low = self.true_action_space.low[:4]
        action_limits_high = self.true_action_space.high[:4]
        # adjust s(pin) angle limits in position control
        if self.action_type == ActionType.POSITION:
            action_limits_low[2] = 90.0
            action_limits_high[2] = 270.0
        # adjust action space
        self.action_space = gym.spaces.Box(
            low=action_limits_low,
            high=action_limits_high,
            dtype=self.true_action_space.dtype,
        )

        self.t_obs = t_obs
        self.time_step_counter = 0
        self.time_limit = time_limit
        self.calculate_accelerations = calculate_accelerations
        self.prev_gripper_state: np.ndarray
        self.prev_gripper_state_vel: np.ndarray
        self.prev_rope_pos: np.ndarray
        self.prev_rope_vel: np.ndarray

        self.with_rgb_images = with_rgb_images
        self.rgb_image_size = rgb_image_size if isinstance(rgb_image_size[0], int) else rgb_image_size[0]
        self.rgb_image_crop_size = rgb_image_crop_size if isinstance(rgb_image_crop_size[0], int) else rgb_image_crop_size[0]

        # TODO: use in reset
        self.seed = seed

        self.observation_buffer = defaultdict(partial(deque, maxlen=self.t_obs))

        scaler_config = OmegaConf.to_container(scaler_config, resolve=True)
        self.normalize_keys: list[str] = scaler_config.get("normalize_keys", [])
        self.standardize_keys: list[str] = scaler_config.get("standardize_keys", [])
        self.normalize_symmetrically: bool = scaler_config["normalize_symmetrically"]
        self.scaler_values: dict[str, dict[str, np.ndarray]] = scaler_config.get("scaler_values", {})

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
        assert all([key in self.extra_scaler_values for key in ["rope_tracking_points"]])

    @property
    def dt(self) -> float:
        """Control frequency of the environment"""
        return self.time_step * self.frame_skip

    @property
    def gripper_state(self) -> np.ndarray:
        tpsd = self.right_gripper.get_articulated_state()[:4].astype(np.float32)
        if "gripper_tpsd_state" in self.normalize_keys:
            tpsd = normalize(tpsd, self.scaler_values["gripper_tpsd_state"], self.normalize_symmetrically)
        elif "gripper_tpsd_state" in self.standardize_keys:
            tpsd = standardize(tpsd, self.scaler_values["gripper_tpsd_state"])
        return tpsd

    @property
    def gripper_state_vel(self) -> np.ndarray:
        if self.time_step_counter == 0:
            tpsd_vel = np.zeros_like(self.gripper_state)
        else:
            tpsd_vel = (self.gripper_state - self.prev_gripper_state) / self.dt
        return tpsd_vel

    @property
    def gripper_state_accel(self) -> np.ndarray:
        if self.time_step_counter == 0:
            tpsd_accel = np.zeros_like(self.gripper_state)
        else:
            tpsd_accel = (self.gripper_state_vel - self.prev_gripper_state_vel) / self.dt
        return tpsd_accel

    @property
    def rope_grasp_index(self) -> int:
        return self.right_gripper.grasp_index_pair[1]

    @property
    def is_rope_grasped(self) -> bool:
        return self.right_gripper.grasp_established

    @property
    def rope_pos(self) -> np.ndarray:
        """Scaled (and flattend) positions of rope points up to and including the grasping point"""
        pos = self.rope.get_positions()[: self.rope_grasp_index + 1].ravel().astype(np.float32)
        if "rope_tracking_points" in self.normalize_keys:
            pos = normalize(
                pos,
                self.scaler_values["rope_tracking_points"],
                self.normalize_symmetrically,
            )
        elif "rope_tracking_points" in self.standardize_keys:
            pos = standardize(
                pos,
                self.scaler_values["rope_tracking_points"],
            )
        return pos

    @property
    def rope_vel(self) -> np.ndarray:
        """Scaled (and flattend) velocities of rope points up to and including the grasping point"""
        if self.time_step_counter == 0:
            vel = np.zeros_like(self.rope_pos)
        else:
            # vel = self.rope.get_velocities()[:self.rope_grasp_index+1].astype(np.float32)
            vel = (self.rope_pos - self.prev_rope_pos) / self.dt
        return vel

    @property
    def rope_accel(self) -> np.ndarray:
        """Scaled (and flattend) accelerations of rope points up to and including the grasping point"""
        if self.time_step_counter == 0:
            accel = np.zeros_like(self.rope_pos)
        else:
            accel = (self.rope_vel - self.prev_rope_vel) / self.dt
        return accel

    @property
    def eyelet(self) -> Eye:
        return self.eyes[0]

    @property
    def eyelet_center_pose(self) -> np.ndarray:
        """Scaled cartesian position and rotation angle [x,y,z,a] of the eyelet's center"""
        eye_xyza = np.concatenate([self.eyelet.center_pose[:3], [self.eyelet.rotation]]).astype(np.float32)
        if "eyelet_center_pose" in self.normalize_keys:
            eye_xyza = normalize(
                eye_xyza,
                self.scaler_values["eyelet_center_pose"],
                self.normalize_symmetrically,
            )
        elif "eyelet_center_pose" in self.standardize_keys:
            eye_xyza = standardize(
                eye_xyza,
                self.scaler_values["eyelet_center_pose"],
            )
        return eye_xyza

    def update_observation_buffer(self) -> None:
        self.observation_buffer["gripper_tpsd_state"].append(deepcopy(self.gripper_state))
        self.observation_buffer["gripper_tpsd_velocity"].append(deepcopy(self.gripper_state_vel))
        self.observation_buffer["gripper_tpsd_state_action"].append(deepcopy(self.gripper_state))
        self.observation_buffer["gripper_tpsd_velocity_action"].append(deepcopy(self.gripper_state_vel))
        self.observation_buffer["eyelet_center_pose"].append(deepcopy(self.eyelet_center_pose))

        rope_positions = deepcopy(self.rope.get_positions()[: self.rope_grasp_index + 1].astype(np.float32))
        self.observation_buffer["rope_tracking_points"].append(rope_positions)
        self.observation_buffer["rope_tracking_points_normalized"].append(
            normalize(
                rope_positions,
                self.extra_scaler_values["rope_tracking_points"],
                self.normalize_symmetrically,
            ).ravel()
        )

        if self.with_rgb_images:
            original_image = self._update_rgb_buffer()
            resized_image = cv2.resize(original_image, self.rgb_image_size)
            cropped_image = resized_image[
                self.rgb_image_size[0] // 2 - self.rgb_image_crop_size[0] // 2 : self.rgb_image_size[0] // 2 + self.rgb_image_crop_size[0] // 2,
                self.rgb_image_size[1] // 2 - self.rgb_image_crop_size[1] // 2 : self.rgb_image_size[1] // 2 + self.rgb_image_crop_size[1] // 2,
            ]
            self.observation_buffer["rgb"].append(np.moveaxis(cropped_image, -1, 0).astype(np.float32) / 255.0)

    def get_observation_dict(self) -> dict[str, np.ndarray]:
        return deque_to_array(self.observation_buffer)

    def step(self, action: np.ndarray):
        self.time_step_counter += 1
        self.prev_gripper_state = self.gripper_state.copy()
        self.prev_gripper_state_vel = self.gripper_state_vel.copy()
        self.prev_rope_pos = self.rope_pos.copy()
        self.prev_rope_vel = self.rope_vel.copy()

        match self.action_type:
            case ActionType.CONTINUOUS:
                key_suffix = None  # actions already scaled to [-1, 1]
            case ActionType.POSITION:
                key_suffix = "_state"
            case ActionType.VELOCITY:
                key_suffix = "_velocity"
            case _:
                raise NotImplementedError

        if (key := f"gripper_tpsd{key_suffix}") in self.normalize_keys:
            action = denormalize(action, self.scaler_values[key], symmetric=self.normalize_symmetrically)
        elif (key := f"gripper_tpsd{key_suffix}") in self.standardize_keys:
            action = destandardize(action, self.scaler_values[key])

        # clip actions to valid tpsd range
        action = np.clip(action, self.action_space.low, self.action_space.high)
        # add gripper state
        ptsda = np.concatenate([action, [5]])
        observation, reward, terminated, truncated, info = super().step(ptsda)

        # add gripper position to info
        info["tool_pose"] = self.right_gripper.get_pose().astype(np.float32)
        if self.calculate_accelerations:
            info["rope_acceleration"] = np.linalg.norm(self.rope_accel.reshape(-1, 3), axis=1).mean()
            info["tool_acceleration"] = self.gripper_state_accel

        # Update the observation buffer
        self.update_observation_buffer()

        # env is truncated when we lost grip of the rope
        truncated = truncated or not self.is_rope_grasped

        # Check if the time limit has been reached
        if self.time_limit is not None:
            if self.time_step_counter >= self.time_limit:
                truncated = True

        return observation, reward, terminated, truncated, info

    def reset(self, *args, **kwargs):
        obs, info = super().reset(*args, **kwargs)
        self.time_step_counter = 0

        # Fill observation buffer with initial values
        for _ in range(self.t_obs):
            self.update_observation_buffer()

        # add gripper position to info
        info["tool_pose"] = self.right_gripper.get_pose().astype(np.float32)
        if self.calculate_accelerations:
            info["rope_acceleration"] = np.linalg.norm(self.rope_accel.reshape(-1, 3), axis=1).mean()
            info["tool_acceleration"] = self.gripper_state_accel
        return obs, info

    def get_rope_points(self) -> np.ndarray:
        """Unscaled positions of rope points up to and including the grasping point"""
        return self.rope.get_positions()[: self.rope_grasp_index + 1].astype(np.float32)

    def _get_distance_to_eyelet(self, rope_point_idx: int) -> float:
        """Return value is negative distance if rope point already passed through eyelet"""
        rope_point = self.get_rope_points()[rope_point_idx]
        sign = self.eyelet.points_are_right_of_eye(rope_point)
        return sign * np.linalg.norm(rope_point - self.eyelet.center_pose[:3])

    def get_distance_rope_tip_eyelet(self) -> float:
        return self._get_distance_to_eyelet(rope_point_idx=0)

    def get_num_rope_points_passed_eyelet(self) -> int:
        """Number of rope points up to and including the grasp point that have passed through the eyelet"""
        rope_points = self.get_rope_points()
        dist_to_eyelet = np.linalg.norm(rope_points - self.rope.sphere_roi_center[0], axis=1)
        indices_near_eyelet = np.flatnonzero(dist_to_eyelet < self.rope.sphere_radius)
        if len(indices_near_eyelet) > 0:
            eyelet_sides = self.eyelet.points_are_right_of_eye(rope_points[indices_near_eyelet])
            rope_is_on_both_sides = eyelet_sides[0] < 0 and eyelet_sides[-1] > 0
            if rope_is_on_both_sides:
                return np.count_nonzero(eyelet_sides >= 0)
        return 0

    def get_fraction_rope_points_passed_eyelet(self) -> float:
        """Fraction of rope points up to and including the grasp point that have passed through the eyelet"""
        # divide by rope_grasp_index (instead of rope_grasp_index + 1)
        # since grasp point on rope will never pass through eyelet in this scenario
        return max(0.0, self.get_num_rope_points_passed_eyelet() / self.rope_grasp_index)


if __name__ == "__main__":
    import itertools

    env = RopeThreadingEnvCompatLayer(
        scaler_config=DictConfig(
            {
                "normalize_keys": [
                    "gripper_tpsd_state",
                    "eyelet_center_pose",
                    "rope_tracking_points",
                ],
                "normalize_symmetrically": True,
                "scaler_values": {
                    "gripper_tpsd_state": {
                        "min": [-90.0, -90.0, 90.0, 0.0],
                        "max": [90.0, 90.0, 270.0, 100.0],
                    },
                    "eyelet_center_pose": {
                        "min": [-25.0, -60.0, -15.0, 55.0],
                        "max": [230.0, 180.0, 135.0, 125.0],
                    },
                    "rope_tracking_points": {
                        "min": 16 * [-25.0, -60.0, -15.0],
                        "max": 16 * [230.0, 180.0, 135.0],
                    },
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
        calculate_accelerations=True,
    )

    seed_list = itertools.cycle([42, 1337, 0, 1, 99])
    env.reset(seed=next(seed_list))

    counter = 0
    while True:
        print(f"{counter}:")
        print(f"\t rope tip to eyelet distance: {env.get_distance_rope_tip_eyelet()}")
        print(f"\t num. rope points passed eyelet: {env.get_num_rope_points_passed_eyelet()}")
        print(f"\t frac. rope points passed eyelet: {env.get_fraction_rope_points_passed_eyelet()}")
        print("\n")

        action = env.action_space.sample()
        normalized_action = normalize(action, env.scaler_values["gripper_tpsd_state"], symmetric=True)
        observation, reward, terminated, truncated, info = env.step(normalized_action)
        done = terminated or truncated
        env.render()

        counter += 1
        if done or (counter % 100 == 0):
            env.reset(seed=next(seed_list))
