from pathlib import Path
from typing import Optional

import numpy as np
import numpy.typing as npt

import pybullet
from pybullet_utils.bullet_client import BulletClient

from movement_primitive_diffusion.workspaces.obstacle_avoidance.pybullet.utils import (
    Body,
    BodyPart,
    Joint,
)

XARM_URDF: Path = Path(__file__).parent / "assets" / "xarm7_with_gripper.urdf"
XARM_GRIPPER_URDF = Path(__file__).parent / "assets" / "xarm_gripper.urdf"
XARM_EEF_LINK = "link_tcp"


class XArmGripper:
    def __init__(
        self,
        robot,
        model_path: Path = XARM_GRIPPER_URDF,
        init_finger_width: Optional[float] = 0.0,
        fix_set_width: bool = False,
    ):
        self._pseudo_ik = None
        self.num_joints = 6
        self.robot = robot
        self.model_path = model_path
        self.init_finger_width = init_finger_width
        self.fix_set_width = fix_set_width
        self._null_space_config = {
            "lowerLimits": self.num_joints * [0],
            "upperLimits": self.num_joints * [0.85],
            "jointRanges": self.num_joints * [0.85],
            "restPoses": self.num_joints * [0.0],
        }
        self.left_finger_joint_names = [
            "drive_joint",
            "left_inner_knuckle_joint",
            "left_finger_joint",
        ]
        self.right_finger_joint_names = [
            "right_outer_knuckle_joint",
            "right_inner_knuckle_joint",
            "right_finger_joint",
        ]

    @property
    def tcp(self) -> BodyPart:
        return self.robot.body.parts["link_tcp"]

    @property
    def left_finger_tcp(self) -> BodyPart:
        return self.robot.body.parts["left_finger_tcp"]

    @property
    def right_finger_tcp(self) -> BodyPart:
        return self.robot.body.parts["right_finger_tcp"]

    @property
    def drive_joint(self) -> Joint:
        return self.robot.body.joints["drive_joint"]

    @property
    def joint_indices(self) -> list[int]:
        return [
            self.robot.body.joints[j].index
            for j in self.left_finger_joint_names + self.right_finger_joint_names
        ]

    def get_joint_values(self) -> tuple[np.ndarray, np.ndarray]:
        return self.robot.get_joint_values(self, self.joint_indices)

    def reset_joints(self, drive_joint_value: float):
        self.robot.reset_joints(
            positions=self.num_joints * [drive_joint_value],
            velocities=self.num_joints * [0],
            joint_indices=self.joint_indices,
        )
        if self.fix_set_width:
            self.move_joints(drive_joint_value)

    def move_joints(
        self, drive_joint_value: float, drive_joint_velocity: Optional[float] = None
    ):
        self.robot._set_joint_motor_controls(
            joint_indices=self.joint_indices,
            control_mode=pybullet.POSITION_CONTROL,
            joint_positions=self.num_joints * [drive_joint_value],
            joint_velocities=None
            if drive_joint_velocity is None
            else self.num_joints * [drive_joint_velocity],
        )

    def set_fingers(self, width: float):
        self.reset_joints(self._ik(width))

    def move_fingers(self, width: float, velocity: Optional[float] = None):
        self.move_joints(self._ik(width), velocity)

    def reset_fingers(self):
        if self.init_finger_width is not None:
            self.set_fingers(self.init_finger_width)
        else:
            self.reset_joints(0)

    def finger_gap_left(self) -> float:
        return self.left_finger_tcp.link_pos()[1] - self.tcp.link_pos()[1]

    def finger_gap_right(self) -> float:
        return self.tcp.link_pos()[1] - self.right_finger_tcp.link_pos()[1]

    # def _ik(self, finger_gap: float, max_iterations: int = 1000) -> float:
    #    # Use separate model to disable joints for inverse kinematic solution
    #    # https://github.com/bulletphysics/bullet3/issues/1626
    #    p = BulletClient(connection_mode=pybullet.DIRECT)
    #    gripper = Body.from_urdf(
    #        bullet_client=p,
    #        file_path=self.model_path,
    #        fixed_base=True,
    #    )
    #    tcp = gripper.parts["link_tcp"]
    #    finger = gripper.parts["left_finger_tcp"]
    #    drive_joint = gripper.joints["drive_joint"]
    #    world_to_finger = p.invertTransform(finger.frame_pos(), finger.frame_orn())
    #    tcp_y = p.multiplyTransforms(
    #        world_to_finger[0],
    #        world_to_finger[1],
    #        tcp.frame_pos(),
    #        tcp.frame_orn(),
    #    )[0][1]
    #    dy = tcp_y - finger_gap
    #    # How to determine dz from dy??? Their relationship is not linear.
    #    # Setting dz to zero will result in a subotimal drive joint value.
    #    dz = 0
    #    position = p.multiplyTransforms(
    #        finger.frame_pos(),
    #        finger.frame_orn(),
    #        [0, dy, dz],
    #        [0, 0, 0, 1],
    #    )[0]
    #    joint_values = p.calculateInverseKinematics(
    #        bodyUniqueId=gripper.id,
    #        endEffectorLinkIndex=finger.index,
    #        targetPosition=position,
    #        #targetOrientation=[0, 0, 0, 1],
    #        maxNumIterations=max_iterations,
    #        residualThreshold=1e-6,
    #        **self._null_space_config,
    #        #currentPositions=self.num_joints * [0],
    #    )
    #    #print(f"{joint_values = }")
    #    drive_joint_value = np.clip(
    #        joint_values[0],
    #        drive_joint.lower_limit,
    #        drive_joint.upper_limit,
    #    )
    #    p.disconnect()
    #    return drive_joint_value

    def _ik(self, finger_width: float, sampling_steps: int = 1_000) -> float:
        if self._pseudo_ik is None:
            # Use separate model to disable joints for inverse kinematic solution
            # https://github.com/bulletphysics/bullet3/issues/1626
            p = BulletClient(connection_mode=pybullet.DIRECT)
            gripper = Body.from_urdf(
                bullet_client=p,
                file_path=self.model_path,
                fixed_base=True,
            )
            tcp = gripper.parts["link_tcp"]
            finger = gripper.parts["left_finger_tcp"]
            driver = gripper.joints["drive_joint"]
            driver_values = []
            dy_values = []
            for i in range(sampling_steps + 1):
                j_val = driver.lower_limit + (i / sampling_steps) * (
                    driver.upper_limit - driver.lower_limit
                )
                for j in [gripper.joints[k] for k in self.left_finger_joint_names]:
                    j.set_state(j_val, velocity=0.0)
                driver_values.append(j_val)
                dy_values.append(finger.link_pos()[1] - tcp.link_pos()[1])
            p.disconnect()
            # flip to sort in ascending order wrt dy_values for np.interp
            dy_values = np.flip(dy_values)
            driver_values = np.flip(driver_values)
            self._pseudo_ik = lambda finger_width: np.interp(
                finger_width, dy_values, driver_values
            )
            # import matplotlib.pyplot as plt
            # plt.plot(driver_values, dy_values)
            # plt.show()
        return self._pseudo_ik(finger_width)


class XArm7:
    def __init__(
        self,
        model_urdf: Path = XARM_URDF,
        eef_name: str = XARM_EEF_LINK,
        init_eef_pos: Optional[npt.ArrayLike] = (0.2, 0, 0.2),
        init_eef_orn: Optional[npt.ArrayLike] = (0, 1, 0, 0),
        max_ik_iterations: int = 500,
        use_null_space_ik: bool = False,
        num_robot_joints: int = 7,
        has_gripper: bool = True,
        init_gripper_fingers_width: Optional[float] = 0.01,
        fix_gripper: bool = True,
    ) -> None:
        self._body = None
        self._robot_joint_indices: list[int] = []
        self._joint_indices: list[int] = []
        self._null_space_config: dict[str, list[float]] = {}

        self.model_urdf: Path = model_urdf
        self.num_robot_joints: int = num_robot_joints
        self.eef_name = eef_name

        self._init_eef_pose = None
        if init_eef_pos is not None or init_eef_orn is not None:
            init_orn = (
                init_eef_orn if init_eef_orn is not None else np.array([0, 0, 0, 1])
            )

            init_pos = init_eef_pos if init_eef_pos is not None else np.zeros(3)
            self._init_eef_pose = np.concatenate([init_pos, init_orn])
        self.max_ik_iterations = max_ik_iterations
        self.use_null_space_ik = use_null_space_ik

        if has_gripper:
            self._gripper = XArmGripper(
                robot=self,
                init_finger_width=init_gripper_fingers_width,
                fix_set_width=fix_gripper,
            )
        else:
            self._gripper = None

    def has_gripper(self) -> bool:
        return self._gripper is not None

    def get_init_eef_pose(self) -> np.ndarray:
        if self._init_eef_pose is not None:
            return self._init_eef_pose
        elif self._p is not None:
            return np.concatenate([self.eef.init_pos, self.eef.init_orn])
        else:
            raise RuntimeError("XArm not initialized.")

    @property
    def num_joints(self) -> int:
        if self.has_gripper():
            return self.num_robot_joints + self.gripper.num_joints
        else:
            return self.num_robot_joints

    @property
    def gripper(self) -> XArmGripper:
        if self._gripper is None:
            raise RuntimeError("Gripper not attached.")
        return self._gripper

    def add_to_scene(
        self,
        bullet_client: BulletClient,
        base_pos: Optional[npt.ArrayLike] = (0, 0, 0),
        base_orn: Optional[npt.ArrayLike] = (0, 0, 0, 1),
        fixed_base: bool = True,
        self_collision: bool = False,
    ) -> int:
        self._p = bullet_client

        loader_flags = pybullet.URDF_GOOGLEY_UNDEFINED_COLORS
        if self_collision:
            loader_flags |= pybullet.URDF_USE_SELF_COLLISION

        self._body = Body.from_urdf(
            bullet_client=self._p,
            file_path=self.model_urdf,
            base_position=base_pos,
            base_orientation=base_orn,
            fixed_base=fixed_base,
            flags=loader_flags,
        )

        assert (
            len(self.body.motor_joints) == self.num_joints
        ), f"Wrong number of motor joints: Expected {self.num_joints}, got {len(self.body.motor_joints)}"
        self._robot_joint_indices = [
            joint.index for joint in self.body.motor_joints[: self.num_robot_joints]
        ]

        robot_joint_limits = self.get_joint_limits()
        self._null_space_config = {
            "lowerLimits": robot_joint_limits[:, 0].tolist(),
            "upperLimits": robot_joint_limits[:, 1].tolist(),
            "jointRanges": (
                robot_joint_limits[:, 1] - robot_joint_limits[:, 0]
            ).tolist(),
            "restPoses": [0.0] * self.num_robot_joints,
        }
        if self.has_gripper():
            for k in self._null_space_config:
                self._null_space_config[k] += self.gripper._null_space_config[k]

        self.reset_eef()
        if self.has_gripper():
            self.gripper.reset_fingers()

        return self.body.id

    def _set_joint_motor_controls(
        self,
        joint_indices: list[int],
        control_mode: int,
        joint_positions: Optional[npt.ArrayLike] = None,
        joint_velocities: Optional[npt.ArrayLike] = None,
        forces: Optional[npt.ArrayLike] = None,
        position_gains: Optional[npt.ArrayLike] = None,
        velocity_gains: Optional[npt.ArrayLike] = None,
        max_velocities: Optional[npt.ArrayLike] = None,
    ):
        kwargs = {
            k: v
            for k, v in dict(
                targetPositions=joint_positions,
                targetVelocities=joint_velocities,
                forces=forces,
                positionGains=position_gains,
                velocityGains=velocity_gains,
                maxVelocities=max_velocities,
            ).items()
            if v is not None
        }
        # setJointMotorControlMultiDofArray
        self._p.setJointMotorControlArray(
            bodyUniqueId=self.body.id,
            jointIndices=joint_indices,
            controlMode=control_mode,
            **kwargs,
        )

    def move_joints(
        self,
        joint_positions: npt.ArrayLike,
        joint_velocities: Optional[npt.ArrayLike] = None,
        max_motor_forces: Optional[npt.ArrayLike] = None,
        position_gains: Optional[npt.ArrayLike] = None,
        velocity_gains: Optional[npt.ArrayLike] = None,
        max_velocities: Optional[npt.ArrayLike] = None,
    ):
        self._set_joint_motor_controls(
            joint_indices=self._robot_joint_indices,
            control_mode=pybullet.POSITION_CONTROL,
            joint_positions=joint_positions,
            joint_velocities=joint_velocities,
            forces=max_motor_forces,
            position_gains=position_gains,
            velocity_gains=velocity_gains,
            max_velocities=max_velocities,
        )

    def set_joint_velocities(
        self,
        joint_velocities: npt.ArrayLike,
        max_motor_forces: Optional[npt.ArrayLike] = None,
    ):
        self._set_joint_motor_controls(
            joint_indices=self._robot_joint_indices,
            control_mode=pybullet.VELOCITY_CONTROL,
            joint_velocities=joint_velocities,
            forces=max_motor_forces,
        )

    def set_joint_torques(self, joint_torques: npt.ArrayLike):
        self._set_joint_motor_controls(
            joint_indices=self._robot_joint_indices,
            control_mode=pybullet.TORQUE_CONTROL,
            forces=joint_torques,
        )

    def reset_joints(
        self,
        positions: npt.ArrayLike,
        velocities: Optional[npt.ArrayLike] = None,
        joint_indices: Optional[list[int]] = None,
    ):
        # https://github.com/bulletphysics/bullet3/issues/2803#issuecomment-770206176
        # def reshape(a: npt.ArrayLike) -> np.ndarray:
        #    x = np.array(a)
        #    return x if x.ndim > 1 else np.reshape(x, (-1, 1))
        # pos = reshape(positions)
        kwargs = (
            {}
            if velocities is None
            else {"targetVelocities": np.atleast_2d(velocities).T.tolist()}
        )
        self._p.resetJointStatesMultiDof(
            bodyUniqueId=self.body.id,
            jointIndices=self._robot_joint_indices
            if joint_indices is None
            else joint_indices,
            targetValues=np.atleast_2d(positions).T.tolist(),
            **kwargs,
        )

    def move_eef(
        self,
        position: npt.ArrayLike,
        orientation: Optional[npt.ArrayLike] = None,
        joint_velocities: Optional[npt.ArrayLike] = None,
        max_motor_forces: Optional[npt.ArrayLike] = None,
    ):
        self.move_joints(
            self.ik(position, orientation),
            joint_velocities=joint_velocities,
            max_motor_forces=max_motor_forces,
        )

    # def set_eef_velocity(self, velocity: npt.ArrayLike):
    #   jacobian = self._p.calculateJacobian(
    #       bodyUniqueId=self.body.id,
    #       linkIndex=self.eef.index,
    #       localPosition=[0]*3,
    #       objPositions=self.get_joint_values()[0].tolist(),
    #       objVelocities=self.get_joint_values()[1].tolist(),
    #       #objPositions=self.ik(self.get_tooltip_pose()[:3], self.get_tooltip_pose()[3:]).tolist(),
    #       #objVelocities=[0]*self.num_joints,
    #       objAccelerations=[0]*self.num_joints,
    #   )
    #   joint_velocities = np.linalg.pinv(jacobian[0]) @ velocity
    #   self.set_joint_velocities(joint_velocities)

    def set_eef(
        self,
        position: npt.ArrayLike,
        orientation: Optional[npt.ArrayLike] = None,
    ):
        self.reset_joints(
            joint_indices=self._robot_joint_indices,
            positions=self.ik(position, orientation),
            velocities=np.zeros(self.num_robot_joints),
        )

    def reset_eef(self):
        eef_pose = self.get_init_eef_pose()
        self.set_eef(eef_pose[:3], eef_pose[3:])

    @property
    def body(self) -> Body:
        if self._body is None:
            raise RuntimeError("Body not initialized.")
        return self._body

    @property
    def eef(self) -> BodyPart:
        return self.body.parts[self.eef_name]

    def get_joint_values(
        self,
        joint_indices: Optional[list[int]] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        # states = np.array([[j.get_position(), j.get_velocity()] for j in self.body.motor_joints]).squeeze()
        # return states[:, 0], states[:, 1]
        states = self._p.getJointStates(
            bodyUniqueId=self.body.id,
            jointIndices=self._robot_joint_indices
            if joint_indices is None
            else joint_indices,
        )
        return np.array([s[0] for s in states]), np.array([s[1] for s in states])

    def ik(
        self,
        position: npt.ArrayLike,
        orientation: Optional[npt.ArrayLike] = None,
    ) -> np.ndarray:
        # By default PyBullet uses the current joint positions of the body.
        # If initial joint positions are provided via the argument 'currentPosition'
        # then targetPosition and targetOrientation are in local space!
        kwargs = {} if orientation is None else {"targetOrientation": orientation}

        if self.use_null_space_ik:
            kwargs |= self._null_space_config

        joint_values = self._p.calculateInverseKinematics(
            bodyUniqueId=self.body.id,
            endEffectorLinkIndex=self.eef.index,
            targetPosition=position,
            maxNumIterations=self.max_ik_iterations,
            residualThreshold=1e-5,
            **kwargs,
        )
        # print(f"{len(joint_values) = }")
        return np.array(joint_values[: self.num_robot_joints])

    def get_joint_by_index(self, i: int) -> Joint:
        joint_index = self._robot_joint_indices[i]
        for joint in self.body.joints.values():
            if joint_index == joint.index:
                return joint
        raise ValueError(f"Joint with index {joint_index} not found.")

    def get_joint_limits(self) -> np.ndarray:
        lower = []
        upper = []
        for i in range(self.num_robot_joints):
            joint = self.get_joint_by_index(i)
            lower.append(joint.lower_limit)
            upper.append(joint.upper_limit)
        return np.row_stack([lower, upper])
