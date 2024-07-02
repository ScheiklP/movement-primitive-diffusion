import numpy as np
from copy import deepcopy
from enum import Enum, unique
from pathlib import Path
from typing import Dict, Optional, Tuple, Union, Callable, List


@unique
class RenderMode(Enum):
    """RenderMode options for SofaEnv.

    This enum specifies if you want to simulate

    - state based without rendering anything (NONE),
    - generate image observations headlessly without creating a window (HEADLESS),
    - generate image observations headlessly, but only when ``env.update_rgb_buffer()`` or ``env.update_rgb_buffer_remote()`` are called manually (MANUAL).
    - create a window to observe the simulation for a remote workstation or WSL (REMOTE).
    - or create a window to observe the simulation (HUMAN).
    """

    HUMAN = 0
    HEADLESS = 1
    REMOTE = 2
    NONE = 3
    MANUAL = 4


@unique
class ObservationType(Enum):
    RGB = 0
    STATE = 1
    RGBD = 2


@unique
class ActionType(Enum):
    DISCRETE = 0
    CONTINUOUS = 1


class BimanualTissueManipulationEnv:
    def __init__(
        self,
        time_step: float = 0.01,
        frame_skip: int = 1,
        scene_path: Union[str, Path] = "scene.py",
        render_mode: RenderMode = RenderMode.HUMAN,
        create_scene_kwargs: Optional[dict] = None,
        on_reset_callbacks: Optional[List[Callable]] = None,
        maximum_state_velocity: Union[np.ndarray, float] = 25.0,
        discrete_action_magnitude: Union[np.ndarray, float] = 40.0,
        image_shape: Tuple[int, int] = (124, 124),
        observation_type: ObservationType = ObservationType.RGB,
        action_type: ActionType = ActionType.CONTINUOUS,
        settle_steps: int = 50,
        reward_amount_dict={
            "sum_distance_markers_to_target": -0.0,
            "sum_delta_distance_markers_to_target": -0.0,
            "sum_markers_at_target": 0.0,
            "left_gripper_workspace_violation": -0.0,
            "right_gripper_workspace_violation": -0.0,
            "left_gripper_state_limit_violation": -0.0,
            "right_gripper_state_limit_violation": -0.0,
            "left_gripper_collsion": -0.0,
            "right_gripper_collsion": -0.0,
            "force_on_tissue": -0.0,
            "successful_task": 0.0,
        },
        individual_agents: bool = False,
        with_collision: bool = True,
        with_tissue_forces: bool = True,
        marker_radius: float = 3.0,
        target_radius: Optional[float] = None,
        show_targets: bool = True,
        render_markers_with_ogl: bool = False,
        num_targets: int = 2,
        random_marker_range: List[Dict[str, Tuple[float, float]]] = [
            {"low": (0.1, 0.3), "high": (0.3, 0.75)},
            {"low": (0.7, 0.3), "high": (0.9, 0.75)},
        ],
        randomize_marker: bool = False,
        random_target_range: List[Dict[str, Tuple[float, float]]] = [
            {"low": (0.0, 0.6), "high": (0.2, 1.2)},
            {"low": (0.8, 0.6), "high": (1.0, 1.2)},
        ],
        randomize_target: bool = True,
    ) -> None:
        self.time_step = time_step
        self.frame_skip = frame_skip
        self.scene_path = scene_path
        self.render_mode = render_mode
        self.create_scene_kwargs = create_scene_kwargs
        self.on_reset_callbacks = on_reset_callbacks
        self.maximum_state_velocity = maximum_state_velocity
        self.discrete_action_magnitude = discrete_action_magnitude
        self.image_shape = image_shape
        self.observation_type = observation_type
        self.action_type = action_type
        self.settle_steps = settle_steps
        self.reward_amount_dict = reward_amount_dict
        self.individual_agents = individual_agents
        self.with_collision = with_collision
        self.with_tissue_forces = with_tissue_forces
        self.marker_radius = marker_radius
        self.target_radius = target_radius
        self.show_targets = show_targets
        self.render_markers_with_ogl = render_markers_with_ogl
        self.num_targets = num_targets
        self.random_marker_range = random_marker_range
        self.randomize_marker = randomize_marker
        self.random_target_range = random_target_range
        self.randomize_target = randomize_target

        self.previous_state = None
        self.time_step_counter = 0

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, dict]:
        self.time_step_counter += 1
        self.previous_pos = deepcopy(self.current_pos)
        return np.random.randn((15)), 0.0, False, self.time_step_counter >= 9, {}

    def reset(self) -> np.ndarray:
        self.time_step_counter = 0
        return np.zeros((0,))

    @property
    def dt(self) -> float:
        """Control frequency of the environment"""
        return self.time_step * self.frame_skip

    @property
    def current_pos(self) -> np.ndarray:
        left_gripper_state = np.random.randn((2))
        right_gripper_state = np.random.randn((2))
        gripper_state = np.concatenate([left_gripper_state, right_gripper_state], axis=-1)

        return gripper_state

    @property
    def current_vel(self) -> np.ndarray:
        if self.time_step_counter == 0:
            vel = np.zeros(self.current_pos.shape)
        else:
            vel = (self.current_pos - self.previous_pos) / self.dt

        return vel

    def render(self) -> np.ndarray:
        return np.random.randn((self.image_shape[0], self.image_shape[1], 3))

    def close(self):
        pass
