from typing import Any, Optional

import os

import numpy as np
import numpy.typing as npt

import gymnasium as gym

import pybullet
from pybullet_utils.bullet_client import BulletClient


try:
    if os.environ["PYBULLET_EGL"]:
        import pkgutil
except:
    pass

DEFAULT_SIZE = 480

# Adapted from:
# https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_envs/env_bases.py


class PyBulletEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        sim_timestep: float = 0.01,
        sim_substeps: int = 1,
        render_mode: Optional[str] = None,
        render_size: tuple[int, int] = (DEFAULT_SIZE, DEFAULT_SIZE),
        default_camera_config={
            "distance": 3,
            "yaw": 0,
            "pitch": -30,
            "lookat": [0, 0, 0],
        },
        num_solver_iterations: int = 5,
        gravity: npt.ArrayLike = [0, 0, -9.81],
        # default_contact_erp: float = 0.9,
    ) -> None:
        self._p: BulletClient = None

        self.timestep = sim_timestep
        self.sim_substeps = sim_substeps
        self.num_solver_iterations = num_solver_iterations
        self.gravity = np.array(gravity)
        # self.default_contact_erp = default_contact_erp

        self.physics_client_id: int = -1
        self.render_mode = render_mode
        self.is_render: bool = render_mode == "human"

        self._render_width: int = render_size[0]
        self._render_height: int = render_size[1]

        self._cam_dist: float = default_camera_config.get("distance", 3)
        self._cam_yaw: float = default_camera_config.get("yaw", 0)
        self._cam_pitch: float = default_camera_config.get("pitch", -30)
        self._cam_lookat: np.ndarray = np.array(
            default_camera_config.get("lookat", [0, 0, 0])
        )

    def _on_reset(self):
        raise (NotImplementedError)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        if self.physics_client_id < 0:
            if self.is_render:
                self._p = BulletClient(connection_mode=pybullet.GUI)
            else:
                self._p = BulletClient(connection_mode=pybullet.DIRECT)
            self._p.resetSimulation()
            self._p.setGravity(self.gravity[0], self.gravity[1], self.gravity[2])
            # self._p.setDefaultContactERP(self.default_contact_erp)
            self._p.setPhysicsEngineParameter(
                fixedTimeStep=self.timestep,  # * self.frame_skip,
                numSolverIterations=self.num_solver_iterations,
                numSubSteps=self.sim_substeps,
                # deterministicOverlappingPairs=1,
            )
            # optionally enable EGL for faster headless rendering
            try:
                if os.environ["PYBULLET_EGL"]:
                    con_mode = self._p.getConnectionInfo()["connectionMethod"]
                    if con_mode == self._p.DIRECT:
                        egl = pkgutil.get_loader("eglRenderer")
                        if egl:
                            self._p.loadPlugin(egl.get_filename(), "_eglRendererPlugin")
                        else:
                            self._p.loadPlugin("eglRendererPlugin")
            except:
                pass
            self.physics_client_id = self._p._client
            self._p.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0)

        self._p.resetDebugVisualizerCamera(
            self._cam_dist, self._cam_yaw, self._cam_pitch, self._cam_lookat
        )

        obs, info = self._on_reset()

        if self.render_mode == "human":
            self.render()

        return obs, info

    def render(self):
        self.is_render = (self.render_mode == "human")

        # if self.physics_client_id >= 0:
        #    self._camera_adjust()

        match self.render_mode:
            case "rgb_array":
                return self._render_rgb_array(
                    img_width=self._render_width,
                    img_height=self._render_height,
                )
            case _:
                return np.array([])

    def _render_rgb_array(
        self,
        img_width: int,
        img_height: int,
    ) -> np.ndarray:
        if self.physics_client_id >= 0:
            view_matrix = self._p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=self._cam_lookat.tolist(),
                distance=self._cam_dist,
                yaw=self._cam_yaw,
                pitch=self._cam_pitch,
                roll=0,
                upAxisIndex=2,
            )
            proj_matrix = self._p.computeProjectionMatrixFOV(
                fov=60,
                aspect=float(img_width) / img_height,
                nearVal=0.1,
                farVal=100.0,
            )
            _, _, px, _, _ = self._p.getCameraImage(
                width=img_width,
                height=img_height,
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix,
                renderer=pybullet.ER_BULLET_HARDWARE_OPENGL,
            )
            self._p.configureDebugVisualizer(
                self._p.COV_ENABLE_SINGLE_STEP_RENDERING, 1
            )
        else:
            px = np.array(
                [[[255, 255, 255, 255]] * img_width] * img_height,
                dtype=np.uint8,
            )

        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(
            np.array(px), (img_height, img_width, -1)
        )
        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def close(self):
        if self.physics_client_id >= 0:
            self._p.disconnect()
        self.physics_client_id = -1
