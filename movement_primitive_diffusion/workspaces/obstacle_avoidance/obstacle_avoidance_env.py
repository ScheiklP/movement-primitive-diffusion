from typing import Any, Optional, Union
from enum import Enum, unique
from collections import deque, defaultdict
from copy import deepcopy
from functools import partial
from pathlib import Path

import numpy as np
import numpy.typing as npt

from omegaconf import DictConfig, OmegaConf

import gymnasium as gym

import pybullet
from pybullet_utils.bullet_client import BulletClient

from movement_primitive_diffusion.utils.helper import deque_to_array
from movement_primitive_diffusion.datasets.scalers import (
    denormalize,
    destandardize,
    normalize,
    standardize,
)
from movement_primitive_diffusion.workspaces.obstacle_avoidance.xarm import XArm7
from movement_primitive_diffusion.workspaces.obstacle_avoidance.pybullet.env import (
    DEFAULT_SIZE,
    PyBulletEnv,
)
from movement_primitive_diffusion.workspaces.obstacle_avoidance.pybullet.utils import Body

ASSETS: Path = (Path(__file__).parent / "assets").resolve()

@unique
class ObservationType(Enum):
    RGB = 0
    STATE = 1
    STATE_RGB = 2

class ObstacleAvoidanceEnv(PyBulletEnv):
    def __init__(
        self,
        scaler_config: DictConfig = DictConfig(
            {
                "normalize_keys": [],
                "standardize_keys": [],
                "normalize_symmetrically": True,
                "scaler_values": {},
            }
        ),
        observation_type: Union[ObservationType, str] = ObservationType.STATE,
        rgb_image_size: Optional[tuple[int, int]] = (DEFAULT_SIZE, DEFAULT_SIZE),
        rgb_image_crop_size: Optional[tuple[int, int]] = None,
        t_obs: int = 10,
        sim_dt: float = 1/1000,
        control_dt: float = 1/30,
        time_limit: Optional[int] = 250,
        workspace_limits_low: npt.ArrayLike = (0.293, -0.3, 0.11),
        workspace_limits_high: npt.ArrayLike = (0.707, 0.38, 0.13),
        render_mode: Optional[str] = "human",
        render_size: tuple[int, int] = (DEFAULT_SIZE, DEFAULT_SIZE),
        camera_config={
            "distance": 1.1,
            "yaw": 90,
            "pitch": -45,
            "lookat": (0.35, 0.04, 0.12),
        },
        max_ik_iterations: int = 1000,
        use_null_space_ik: bool = False,
        seed: Optional[int] = None, # NOTE: Unused
    ) -> None:
        assert 0 < sim_dt <= control_dt, "Simulation dt must be less than or equal to a positive control dt"

        super().__init__(
            sim_timestep=sim_dt,
            sim_substeps=1,
            render_mode=render_mode,
            render_size=render_size,
            default_camera_config=camera_config,
        )
        self.sim_dt = sim_dt
        self.control_dt = control_dt

        self.robot = XArm7(
            init_eef_pos=(0.525, -0.28, 0.12),
            init_eef_orn=(0, 1, 0, 0),
            max_ik_iterations=max_ik_iterations,
            use_null_space_ik=use_null_space_ik,
        )
        self.scene = ObstacleAvoidanceScene()
        # track the turns the robot took at each obstacle level
        self.mode = Mode()

        self.step_count: int = 0
        self.step_limit: Optional[int] = time_limit
        # TODO: use in reset
        #self.seed = seed

        # action space
        self.workspace_limits = np.row_stack([workspace_limits_low, workspace_limits_high]).astype(np.float32)
        # actions are (x, y) cartesian coordinates of the end effector
        self.action_space = gym.spaces.Box(
            low=self.workspace_limits[0, :2],
            high=self.workspace_limits[1, :2],
            shape=(2,),
            dtype=np.float32,
        )

        # observation space
        def make_state_space() -> gym.spaces.Box:
            # observations are (x, y) cartesian coordinates of the end effector
            return gym.spaces.Box(
                low=self.workspace_limits[0, :2],
                high=self.workspace_limits[1, :2],
                shape=(2,),
                dtype=np.float32,
            )

        def make_rgb_space() -> gym.spaces.Box:
            if rgb_image_size is None:
                raise ValueError("RGB observation_type requires rgb_image_size to be specified.")
            if rgb_image_crop_size is None:
                space_shape = rgb_image_size + (3,)
            else:
                space_shape = (
                    (rgb_image_size[0] // 2 + rgb_image_crop_size[0] // 2) - (rgb_image_size[0] // 2 - rgb_image_crop_size[0] // 2),
                    (rgb_image_size[1] // 2 + rgb_image_crop_size[1] // 2) - (rgb_image_size[1] // 2 - rgb_image_crop_size[1] // 2),
                    3,
                )
            if space_shape[0] <= 0 or space_shape[1] <= 0:
                raise ValueError(f"Invalid rgb_image_size and rgb_image_crop_size combination: {rgb_image_size}, {rgb_image_crop_size}")
            return gym.spaces.Box(low=0, high=255, shape=space_shape, dtype=np.uint8)

        if isinstance(observation_type, str):
            self.observation_type = ObservationType[observation_type.upper()]
        else:
            self.observation_type = observation_type

        match self.observation_type:
            case ObservationType.RGB:
                self.observation_space = make_rgb_space()
            case ObservationType.STATE:
                self.observation_space = make_state_space()
            case ObservationType.STATE_RGB:
                self.observation_space = gym.spaces.Dict({"state": make_state_space(), "rgb": make_rgb_space()})
            case _:
                raise ValueError(f"Observation type {observation_type} not supported.")
        self.rgb_image_size = rgb_image_size
        self.rgb_image_crop_size = rgb_image_crop_size
        self.t_obs = t_obs
        self.observation_buffer = defaultdict(partial(deque, maxlen=self.t_obs))

        self.previous_agent_pos = None
        self.latest_action = None
        self.latest_action_vel = np.array([0, 0], dtype=np.float32)
        self.latest_normalized_action_vel = np.array([0, 0], dtype=np.float32)
        self.latest_normalized_action = None

        # SCALING AND NORMALIZATION
        scaler_config = OmegaConf.to_container(scaler_config, resolve=True)
        self.normalize_keys: list[str] = scaler_config.get("normalize_keys", [])
        self.standardize_keys: list[str] = scaler_config.get("standardize_keys", [])
        self.normalize_symmetrically: bool = scaler_config["normalize_symmetrically"]
        self.scaler_values: dict[str, dict[str, np.ndarray]] = scaler_config.get(
            "scaler_values", {}
        )

        for key in self.normalize_keys:
            assert key in self.scaler_values, f"Key {key} not found in scaler values."
            for metric in ["min", "max"]:
                assert (
                    metric in self.scaler_values[key]
                    and self.scaler_values[key][metric] is not None
                ), f"Key {key} does not have {metric} in scaler values."
                self.scaler_values[key][metric] = np.array(self.scaler_values[key][metric], dtype=np.float32)

        for key in self.standardize_keys:
            assert key in self.scaler_values, f"Key {key} not found in scaler values."
            for metric in ["mean", "std"]:
                assert (
                    metric in self.scaler_values[key]
                    and self.scaler_values[key][metric] is not None
                ), f"Key {key} does not have {metric} in scaler values."
                self.scaler_values[key][metric] = np.array(self.scaler_values[key][metric], dtype=np.float32)

    @property
    def timestamp(self) -> float:
        return self.step_count * self.dt

    @property
    def dt(self) -> float:
        # return self.timestep * self.frame_skip
        return self.control_dt

    @property
    def current_agent_pos(self) -> np.ndarray:
        pos = np.array(self.robot.eef.pos()[:2], dtype=np.float32)
        if "agent_pos" in self.normalize_keys:
            pos = normalize(pos, self.scaler_values["agent_pos"], self.normalize_symmetrically)
        if "agent_pos" in self.standardize_keys:
            pos = standardize(pos, self.scaler_values["agent_pos"])
        return pos

    @property
    def current_agent_vel(self) -> np.ndarray:
        if "agent_pos" in self.standardize_keys:
            raise RuntimeError(f"Normalization of agent velocity is not supported on standardized_agent position")
        if self.step_count == 0:
            vel = np.zeros(self.current_agent_pos.shape, dtype=np.float32)
        else:
            vel = (self.current_agent_pos - self.previous_agent_pos) / self.dt
        # if "agent_pos" not in self.normalize_keys:
        #    sim_vel = np.array(self.robot.eef.linear_velocity()[:2], dtype=np.float32)
        #    assert np.allclose(vel, sim_vel), "Computed velocity is not close to simulated velocity"
        return vel

    def current_action_and_vel(self) -> tuple[np.ndarray, np.ndarray]:
        if self.latest_normalized_action is None:
            return self.current_agent_pos, self.current_agent_vel
        assert isinstance(self.latest_normalized_action, np.ndarray), f"{type(self.latest_normalized_action)}"
        assert isinstance(self.latest_normalized_action_vel, np.ndarray), f"{type(self.latest_normalized_action_vel)}"
        return self.latest_normalized_action, self.latest_normalized_action_vel

    def _on_reset(self):
        self.step_count = 0

        self.latest_action = None
        self.latest_action_vel = np.array([0, 0], dtype=np.float32)
        self.latest_normalized_action = None
        self.latest_normalized_action_vel = np.array([0, 0], dtype=np.float32)

        # toggle rendering during scene setup
        self._p.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 0)
        if not self.scene.is_loaded():
            self.scene.load(self._p)
            self.robot.add_to_scene(self._p, base_pos=(0.1, 0, 0))

        self.scene.reset()
        self.mode.reset(self.scene)
        self.robot.reset_eef()
        self.robot.gripper.reset_fingers()
        # self._workspace_aabb = self._compute_workspace_aabb()
        self._p.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 1)

        # disable collision between robot base and lab surrounding
        for part in ["table_plane", "panda_ground", "support_body"]:
            self._p.setCollisionFilterPair(
                bodyUniqueIdA=self.scene.bodies["lab"].id,
                bodyUniqueIdB=self.robot.body.id,
                linkIndexA=self.scene.bodies["lab"].parts[part].index,
                linkIndexB=self.robot.body.parts["link_base"].index,
                enableCollision=0,
            )

        # Fill observation buffer with initial values
        for _ in range(self.t_obs):
            self.update_observation_buffer()

        return self._get_obs(), self._get_info()

    def update_observation_buffer(self, observation: Optional[Union[np.ndarray, dict[str, np.ndarray]]] = None) -> None:
        if observation is None:
            obs = self._get_obs()
        else:
            obs = observation
        match self.observation_type:
            case ObservationType.RGB:
                self.observation_buffer["rgb"].append(np.moveaxis(obs, -1, 0).astype(np.float32) / 255.0)
            case ObservationType.STATE:
                self.observation_buffer["state"].append(obs)
            case ObservationType.STATE_RGB:
                self.observation_buffer["state"].append(obs["state"])
                self.observation_buffer["rgb"].append(np.moveaxis(obs["rgb"], -1, 0).astype(np.float32) / 255.0)
        # agent state
        self.observation_buffer["agent_pos"].append(self.current_agent_pos)
        self.observation_buffer["agent_vel"].append(self.current_agent_vel)
        # action 
        action, action_vel = self.current_action_and_vel()
        self.observation_buffer["action"].append(action)
        self.observation_buffer["action_vel"].append(action_vel)

    def get_observation_dict(self, add_batch_dim: bool = True) -> dict:
        return deque_to_array(self.observation_buffer, add_batch_dim=add_batch_dim)

    def step(self, action):
        self.step_count += 1
        self.previous_agent_pos = deepcopy(self.current_agent_pos)

        if self.latest_normalized_action is not None:
            self.latest_normalized_action_vel = (action - self.latest_normalized_action) / self.dt
        self.latest_normalized_action = deepcopy(action)

        # convert action into (x, y) coordinates of the end effector
        action = self.denormalize_destandardize_actions(action)

        # compute action velocity
        if self.latest_action is not None:
            self.latest_action_vel = (action - self.latest_action) / self.dt
        else:
            self.latest_action_vel = np.zeros(action.shape, dtype=np.float32)
        self.latest_action = deepcopy(action)

        # execute action
        eef_pose = self.robot.get_init_eef_pose()
        eef_pos = np.concatenate([action, [eef_pose[2]]], dtype=eef_pose.dtype)
        # self.robot.set_eef(position=eef_pos, orientation=eef_pose[3:], dtype=eef_pose.dtype)
        self.robot.move_eef(position=eef_pos, orientation=eef_pose[3:])
        self.robot.gripper.reset_fingers()
        # step simulation
        nsteps = int(self.control_dt // self.sim_dt)
        for _ in range(nsteps):
            self._p.stepSimulation()
            # break on robot collision
            #colls = self._p.getContactPoints(bodyA=self.robot.body.id)
            #if len(colls) > 0:
            #    break

        observation = self._get_obs()
        self.update_observation_buffer(observation)

        self.mode.update(self.robot.eef.pos())
        info = self._get_info()

        reward = self._get_reward()

        # terminate the episode if the robot collides with an obstacle or reaches the goal
        terminated = len(info["obstacle_collisions"]) > 0 or info["success"]

        # truncate the episode if the robot collides with anything besides an obstacle in the scene
        # or step limit is reached
        truncated = len(info["robot_collisions"]) > 0
        if self.step_limit is not None:
            if self.step_count >= self.step_limit:
                truncated = True

        return observation, reward, terminated, truncated, info

    def denormalize_destandardize_actions(self, action: np.ndarray) -> np.ndarray:
        action_copy = deepcopy(action)
        if (key := "action") in self.normalize_keys:
            action_copy[:2] = denormalize(action_copy[:2], self.scaler_values[key], symmetric=self.normalize_symmetrically)
        if (key := "action") in self.standardize_keys:
            action_copy[:2] = destandardize(action_copy[:2], self.scaler_values[key])
        action_copy = np.clip(action_copy, self.action_space.low, self.action_space.high)
        return action_copy

    def _get_obs(self) -> Union[np.ndarray, dict[str, np.ndarray]]:
        match self.observation_type:
            case ObservationType.RGB:
                return self._get_rgb_obs()
            case ObservationType.STATE:
                return self._get_state_obs()
            case ObservationType.STATE_RGB:
                return {"state": self._get_state_obs(), "rgb": self._get_rgb_obs()}
            case _:
                raise ValueError(f"Observation type {self.observation_type} not supported.")

    def _get_state_obs(self) -> np.ndarray:
        state_obs = np.array(self.robot.eef.pos()[:2], dtype=np.float32)
        if "state" in self.normalize_keys:
            state_obs = normalize(state_obs, self.scaler_values["state"], self.normalize_symmetrically)
        if "state" in self.standardize_keys:
            state_obs = standardize(state_obs, self.scaler_values["state"])
        return state_obs

    def _get_rgb_obs(self) -> np.ndarray:
        rgb_array = self._render_rgb_array(img_width=self.rgb_image_size[0], img_height=self.rgb_image_size[1])
        if self.rgb_image_crop_size is None:
            return rgb_array
        else:
            return rgb_array[
                self.rgb_image_size[0] // 2 - self.rgb_image_crop_size[0] // 2 : self.rgb_image_size[0] // 2 + self.rgb_image_crop_size[0] // 2,
                self.rgb_image_size[1] // 2 - self.rgb_image_crop_size[1] // 2 : self.rgb_image_size[1] // 2 + self.rgb_image_crop_size[1] // 2,
            ].copy()

    def _get_reward(self) -> float:
        def squared_exp_kernel(eef_pos, obs_pos, scale, bandwidth):
            return scale * np.exp(np.square(np.linalg.norm(eef_pos - obs_pos)) / bandwidth)
        reward = 0.0
        eef_pos = self.robot.eef.pos()
        for obs in self.scene.obstacles.values():
            reward += squared_exp_kernel(eef_pos[:2], obs.pos()[:2], 1, 1)
        # divergence from center of finish line
        reward -= np.abs(eef_pos[0] - self.scene.get_goal()[0])
        return reward

    def _get_info(self) -> dict[str, Any]:
        j_pos, j_vel = self.robot.get_joint_values()
        # check failure condition: robot eef collided with an obstacle?
        obs_collisions, obs_coll_data = self._check_obstacle_collisions()
        coll_with_obs = len(obs_collisions) > 0
        # check truncation condition: robot collided with anything besides an obstacle?
        robot_collisions = self._check_robot_collisions(ignore=obs_coll_data)
        return {
            "env_step": self.step_count,
            "env_timestamp": self.timestamp,
            "agent_pos": self.current_agent_pos,
            "agent_vel": self.current_agent_vel,
            "success": (not coll_with_obs) and self._check_success(),
            "mode": self.mode.get_encoding(),
            "obstacle_collisions": obs_collisions,
            "robot_collisions": robot_collisions,
            "goal_pos": self.scene.get_goal(),
            "eef_pos": self.robot.eef.pos(),
            "eef_orn": self.robot.eef.orn(),
            "eef_vel": self.robot.eef.linear_velocity(),
            "joint_pos": j_pos,
            "joint_vel": j_vel,
        }

    def _check_obstacle_collisions(self) -> tuple[list[list[str]], list[list[int]]]:
        obs_collisions = []
        coll_data = []
        for obs_name, obs in self.scene.obstacles.items():
            colls = self._p.getContactPoints(bodyA=self.robot.body.id, bodyB=obs.id)
            for col in colls:
                if (
                    c := [self.robot.body.get_part_by_index(col[3]).name, obs_name]
                ) not in obs_collisions:
                    obs_collisions.append(c)
                    coll_data.append(col[1:5])
        return obs_collisions, coll_data

    def _check_robot_collisions(self, ignore: list[list[int]]) -> list[list[str]]:
        # check if robot is collided with anything (besides the obstacles) in the scene and fail if so
        robot_collisions = []
        colls = self._p.getContactPoints(bodyA=self.robot.body.id)
        for col in colls:
            if col[1:5] in ignore:
                continue
            name_A, link_A = (
                self.robot.body.name,
                self.robot.body.get_part_by_index(col[3]).name,
            )
            try:
                bodyB = self.scene.get_body_by_id(col[2])
                name_B, link_B = (bodyB.name, bodyB.get_part_by_index(col[4]).name)
            except ValueError:
                name_B, link_B = (f"body_id={col[2]}", f"link_index={col[4]}")
            if (c := [name_A, name_B, link_A, link_B]) not in robot_collisions:
                robot_collisions.append(c)
        return robot_collisions

    def _check_success(self) -> bool:
        return self.robot.eef.pos()[1] > self.scene.get_goal()[1]

    def _compute_workspace_aabb(self) -> np.ndarray:
        # return np.array([[0.293, -0.3, 0.11], [0.707, 0.38, 0.13]])
        dx = 0.03
        dy = 0.02
        dz = 0.01
        eef_pos = self.robot.get_init_eef_pose()[:3]
        aabb = env.scene.get_aabb()
        # extend aabb by eef init pos
        aabb[0] = np.minimum(aabb[0], eef_pos)
        # extend in x
        aabb[0, 0] -= dx
        aabb[1, 0] += dx
        # extend in y
        aabb[0, 1] -= dy
        aabb[1, 1] += dy
        # adjust z
        aabb[0, 2] = eef_pos[2] - dz
        aabb[1, 2] = eef_pos[2] + dz
        return aabb


class ObstacleAvoidanceScene:
    def __init__(
        self,
        lab_path: Path = ASSETS / "lab_surrounding.urdf",
        finish_line_path: Path = ASSETS / "finish_line.urdf",
        obs_0_path: Path = ASSETS / "obstacle_0.urdf",
        obs_1_path: Path = ASSETS / "obstacle_1.urdf",
        init_lab_pos: npt.ArrayLike = [0.4, 0, -0.02],  # [0.2, 0, -0.02]
        obs_lvl_origin: npt.ArrayLike = [0.5, -0.1, 0.0],
        obs_lvl_offset: npt.ArrayLike = [0.075, 0.18],
        finish_line_x: float = 0.4,
    ):
        self.lab_path = lab_path
        self.finish_line_path = finish_line_path
        self.obs_0_path = obs_0_path
        self.obs_1_path = obs_1_path
        self.init_lab_pos = np.array(init_lab_pos)
        self.obs_lvl_origin = np.array(obs_lvl_origin)
        self.obs_lvl_offset = np.array(obs_lvl_offset)
        self.finish_line_x = finish_line_x

        self.bodies: dict[str, Body] = {}
        self.obstacles: dict[str, Body] = {}
        self.trajectories: dict[str, dict[str, Union[int, list[int]]]] = {}

        self._bullet_client = None

    def load(self, bullet_client: BulletClient):
        self._bullet_client = bullet_client
        # lab surrounding
        self.bodies["lab"] = Body.from_urdf(
            bullet_client=bullet_client,
            file_path=self.lab_path,
            base_position=self.init_lab_pos,
            fixed_base=True,
        )
        # add finish line
        finish_pos = self.get_goal().tolist()
        self.bodies["finish_line"] = Body.from_urdf(
            bullet_client=bullet_client,
            file_path=self.finish_line_path,
            base_position=finish_pos,
            fixed_base=True,
        )

        # add obstacles
        def load_obstacle(obs: int, y: int, x: int) -> Body:
            pos = (
                self.obs_lvl_origin[0] + x * y * self.obs_lvl_offset[0],
                self.obs_lvl_origin[1] + y * self.obs_lvl_offset[1],
                self.obs_lvl_origin[2],
            )
            return Body.from_urdf(
                bullet_client=bullet_client,
                file_path={0: self.obs_0_path, 1: self.obs_1_path}[obs],
                base_position=pos,
                fixed_base=True,
            )

        self.obstacles = {
            "obs_l1_mid": load_obstacle(obs=0, y=0, x=0),
            "obs_l2_top": load_obstacle(obs=1, y=1, x=-1),
            "obs_l2_bot": load_obstacle(obs=1, y=1, x=1),
            "obs_l3_top": load_obstacle(obs=1, y=2, x=-1),
            "obs_l3_mid": load_obstacle(obs=1, y=2, x=0),
            "obs_l3_bot": load_obstacle(obs=1, y=2, x=1),
        }
        self.bodies |= self.obstacles

    def get_goal(self) -> np.ndarray:
        #return self.bodies["finish_line"].pos()
        return np.array([
            self.finish_line_x,
            self.obs_lvl_origin[1] + 2.5 * self.obs_lvl_offset[1],
            self.obs_lvl_origin[2],
        ])

    def get_level_positions(self, level: int) -> np.ndarray:
        x = np.linspace(np.fix(-level / 2), np.fix(level / 2), level)
        y = level - 1
        x_pos = self.obs_lvl_origin[0] + x * y * self.obs_lvl_offset[0]
        y_pos = np.full(level, self.obs_lvl_origin[1] + y * self.obs_lvl_offset[1])
        z_pos = np.full(level, self.obs_lvl_origin[2])
        return np.column_stack([x_pos, y_pos, z_pos])

    def get_level_extends(self, level: int) -> np.ndarray:
        extend = []
        for obs_name in self.obstacles:
            if obs_name.startswith(f"obs_l{level}"):
                aabb = self.obstacles[obs_name].aabb()
                extend.append(aabb[1, :] - aabb[0, :])
        return np.array(extend)

    def is_loaded(self) -> bool:
        return self._bullet_client is not None

    @property
    def bullet_client(self) -> BulletClient:
        if self._bullet_client is not None:
            return self._bullet_client
        else:
            raise RuntimeError("Scene is not loaded yet!")

    def reset(self):
        # clear trajectories
        self.bullet_client.removeAllUserDebugItems()
        # for name in self.trajectories.keys():
        #    self.remove_trajectory(name, False)
        self.trajectories.clear()
        # reset bodies
        for body in self.bodies.values():
            body.reset_to_initial_pose()

    def get_body_by_id(self, id: int) -> Body:
        for body in self.bodies.values():
            if body.id == id:
                return body
        raise ValueError(f"Body with id {id} not found!")

    def get_aabb(self) -> np.ndarray:
        # return np.array([
        #    [0.23264376, -0.28074502, 0.0016839],
        #    [0.75710598, 0.48882787, 0.12580657]
        # ])
        # get aabb of all obstacles
        aabb = np.stack([np.full(3, np.inf), np.full(3, -np.inf)])
        for obs in self.obstacles.values():
            obs_aabb = obs.aabb()
            aabb[0, :] = np.minimum(aabb[0, :], obs_aabb[0, :])
            aabb[1, :] = np.maximum(aabb[1, :], obs_aabb[1, :])
        # restrict height to table plane
        table_aabb = self.bodies["lab"].parts["table_plane"].aabb()
        aabb[0, 2] = np.maximum(aabb[0, 2], table_aabb[1, 2])
        # extend width by finish line
        finish_aabb = self.bodies["finish_line"].aabb()
        aabb[1, 1] = np.maximum(aabb[1, 1], finish_aabb[1, 1])
        return aabb

    def show_aabb(
        self,
        aabb: np.ndarray,
        color=[0.5, 0.5, 0],
        line_width: float = 1.0,
    ):
        def vertex(x, y, z):
            return [aabb[x, 0], aabb[y, 1], aabb[z, 2]]

        def adjacent(x, y, z):
            return [vertex(1 - x, y, z), vertex(x, 1 - y, z), vertex(x, y, 1 - z)]

        self.bullet_client.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 0)
        for x, y, z in [(0, 0, 0), (1, 1, 0), (1, 0, 1), (0, 1, 1)]:
            p = vertex(x, y, z)
            for q in adjacent(x, y, z):
                self.bullet_client.addUserDebugLine(
                    p, q, lineColorRGB=color, lineWidth=line_width
                )
        self.bullet_client.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 1)

    def add_trajectory(
        self,
        name: str,
        points: np.ndarray,
        color=[0, 1, 0],
    ):
        self.bullet_client.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 0)
        if name in self.trajectories:
            self.remove_trajectory(name)
        if name not in self.trajectories:
            self.trajectories[name] = {"waypoints": -1, "segments": []}
        # add line segments
        for i in range(points.shape[0] - 1):
            id = self.bullet_client.addUserDebugLine(
                lineFromXYZ=points[i],
                lineToXYZ=points[i + 1],
                lineColorRGB=color,
                lineWidth=1.0,
            )
            if id > -1:
                self.trajectories[name]["segments"].append(id)
        self.bullet_client.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 1)

    def remove_trajectory(
        self,
        name: str,
        drop: bool = True,
    ):
        self.bullet_client.removeUserDebugItem(
            itemUniqueId=self.trajectories[name]["waypoints"]
        )
        for id in self.trajectories[name]["segments"]:
            self.bullet_client.removeUserDebugItem(itemUniqueId=id)
        if drop:
            self.trajectories.pop(name)


class Mode:
    def __init__(self):
        self.encoding = np.zeros(2 + 3 + 4)
        self.l1_passed = False
        self.l2_passed = False
        self.l3_passed = False

    @classmethod
    def from_encoding(cls, encoding: np.ndarray):
        assert encoding.shape == (
            2 + 3 + 4,
        ), f"invalid encoding shape: {encoding.shape}"
        assert np.sum(encoding) <= 3, f"invalid encoding: {encoding}"
        mode = cls()
        mode.encoding = np.copy(encoding)
        mode.l1_passed = np.any(mode.encoding[0:2] == 1)
        mode.l2_passed = np.any(mode.encoding[2:5] == 1)
        mode.l3_passed = np.any(mode.encoding[5:9] == 1)
        return mode

    def reset(self, scene: ObstacleAvoidanceScene):
        assert scene.is_loaded(), "scene not loaded"

        self.l1_passed = False
        self.l2_passed = False
        self.l3_passed = False
        assert np.sum(self.encoding) <= 3
        self.encoding = np.zeros(2 + 3 + 4)

        # self.obs_half_extend_y = 0.5 * scene.get_level_extends(1)[0, 1]
        # assert (self.obs_half_extend_y == 0.03), f"obs_half_extend_y error: {self.obs_half_extend_y}"
        self.obs_half_extend_y = 0.03

        level_pos = [scene.get_level_positions(i) for i in range(1, 4)]
        self.l1_y = level_pos[0][0, 1]
        self.l2_y = level_pos[1][0, 1]
        self.l3_y = level_pos[2][0, 1]
        # level 1
        self.l1_mid_x = level_pos[0][0, 0]
        # level 2
        self.l2_top_x = level_pos[1][0, 0]
        self.l2_bot_x = level_pos[1][1, 0]
        # level 3
        self.l3_top_x = level_pos[2][0, 0]
        self.l3_mid_x = level_pos[2][1, 0]
        self.l3_bot_x = level_pos[2][2, 0]

    def get_encoding(self) -> np.ndarray:
        return np.copy(self.encoding)

    def update(self, eef_position: np.ndarray):
        eef_x, eef_y = eef_position[:2]
        # level 1
        if not self.l1_passed:
            if abs(eef_y - self.l1_y) <= self.obs_half_extend_y:
                # assert (eef_y - self.obs_half_extend_y <= self.l1_y <= eef_y + self.obs_half_extend_y), "alignment error level 1"
                if eef_x < self.l1_mid_x:
                    self.encoding[0] = 1
                elif eef_x > self.l1_mid_x:
                    self.encoding[1] = 1
                self.l1_passed = True
        # level 2
        if not self.l2_passed:
            if abs(eef_y - self.l2_y) <= self.obs_half_extend_y:
                # assert (eef_y - 0.03 <= self.l2_y <= eef_y + 0.03), "alignment error level 2"
                if eef_x < self.l2_top_x:
                    self.encoding[2] = 1
                elif self.l2_top_x < eef_x < self.l2_bot_x:
                    self.encoding[3] = 1
                elif eef_x > self.l2_bot_x:
                    self.encoding[4] = 1
                self.l2_passed = True
        # level 3
        if not self.l3_passed:
            if eef_y >= self.l3_y:
                if eef_x < self.l3_top_x:
                    self.encoding[5] = 1
                if self.l3_top_x < eef_x < self.l3_mid_x:
                    self.encoding[6] = 1
                elif self.l3_mid_x < eef_x < self.l3_bot_x:
                    self.encoding[7] = 1
                elif eef_x > self.l3_top_x:
                    self.encoding[8] = 1
                self.l3_passed = True

    def decode(self) -> float:
        """decode to decimal number"""
        return self.encoding.dot(1 << np.arange(self.encoding.shape[-1]))

    @staticmethod
    def compute_distribution(modes: list["Mode"]) -> tuple[np.ndarray, float]:
        if len(modes) > 0:
            _, counts = np.unique([mode.decode() for mode in modes], return_counts=True)
            mode_dist = counts / np.sum(counts)
            entropy = -np.sum(mode_dist * (np.log(mode_dist) / np.log(24)))
        else:
            counts = np.zeros(1)
            entropy = 0.0
        return counts, entropy


if __name__ == "__main__":
    import time
    from pathlib import Path
    import git
    import torch
    from omegaconf import DictConfig
    import cv2
    
    from movement_primitive_diffusion.datasets.trajectory_dataset import TrajectoryDataset

    dataset_hz = 30

    git_repo = git.Repo(".", search_parent_directories=True)
    git_root = git_repo.git.rev_parse("--show-toplevel")
    trajectories_root = Path(git_root) / "data" / f"obstacle_avoidance_trajectories_{dataset_hz}_hz"

    dataset = TrajectoryDataset(
        trajectory_dirs=[path for path in trajectories_root.iterdir() if path.is_dir()],
        keys=["action", "agent_pos", "agent_vel"],
        dt=1.0 / dataset_hz,
        normalize_keys=["action", "agent_pos"],
        normalize_symmetrically=True,
        scaler_values={
            "action": {
                "min": torch.tensor([0.293, -0.3]),
                "max": torch.tensor([0.707, 0.38]),
            },
            "agent_pos": {
                "min": torch.tensor([0.293, -0.3]),
                "max": torch.tensor([0.707, 0.38]),
            },
        },
        calculate_velocities_from_to=[
            ("agent_pos", "agent_vel"),
            ("action", "action_vel"),
        ],
        recalculate_velocities_from_to=[
            ("agent_pos", "agent_vel"),
            ("action", "action_vel"),
        ],
    )

    action_scaler = DictConfig(
        {
            "normalize_keys": ["action"],
            "standardize_keys": [],
            "normalize_symmetrically": True,
            "scaler_values": {
                "action": {
                    "min": [0.293, -0.3],
                    "max": [0.707, 0.38],
                },
            },
        }
    )

    env = ObstacleAvoidanceEnv(
        scaler_config=action_scaler,
        time_limit=3500,
        control_dt=1.0 / dataset_hz,
        render_mode=None,
        observation_type="rgb",
        camera_config={
            "distance": 0.9,
            "yaw": 90,
            "pitch": -45,
            "lookat": (0.35, 0.04, 0.12),
        },
        rgb_image_size=(256, 256),
        #rgb_image_crop_size=(96, 96),
    )

    for i in range(len(dataset)):
        trajectory = dataset[i]
        obs, info = env.reset()
        for idx in range(trajectory["action"].shape[0]):
            action = trajectory["action"][idx].detach().cpu().numpy()
            obs, reward, terminated, truncated, info = env.step(action)
            if env.observation_type in [ObservationType.RGB, ObservationType.STATE_RGB]:
                cv2.imshow("Obstacle Avoidance", obs[..., ::-1])
                cv2.waitKey(1)
            if terminated or truncated:
                break
            #time.sleep(env.dt)