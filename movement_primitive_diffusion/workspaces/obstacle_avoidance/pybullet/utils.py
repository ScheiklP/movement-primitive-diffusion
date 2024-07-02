from dataclasses import dataclass, field, InitVar

from pathlib import Path
from typing import Optional

import numpy as np
import numpy.typing as npt

import pybullet
from pybullet_utils.bullet_client import BulletClient

# Adapted from:
# https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_envs/robot_bases.py


@dataclass
class Joint:
    bullet_client: BulletClient
    body_id: int
    index: int
    name: str = field(init=False)
    joint_type: int = field(init=False)
    lower_limit: float = field(init=False)
    upper_limit: float = field(init=False)
    link_name: str = field(init=False)

    def __post_init__(self):
        joint_info = self.bullet_client.getJointInfo(self.body_id, self.index)
        self.name = joint_info[1].decode("utf-8")
        self.joint_type = joint_info[2]
        self.lower_limit = joint_info[8]
        self.upper_limit = joint_info[9]
        self.link_name = joint_info[12].decode("utf-8")

    def get_state(self) -> tuple[np.ndarray, np.ndarray]:
        x, vx, _, _ = self.bullet_client.getJointStateMultiDof(self.body_id, self.index)
        return np.array(x), np.array(vx)

    def get_position(self) -> np.ndarray:
        x, _ = self.get_state()
        return x

    # def current_relative_position(self) -> tuple[np.ndarray, np.ndarray]:
    #    pos, vel = self.get_state()
    #    pos_mid = 0.5 * (self.lower_limit + self.upper_limit)
    #    return 2 * (pos - pos_mid) / (self.upper_limit - self.lower_limit), 0.1 * vel

    def get_orientation(self) -> np.ndarray:
        _, r = self.get_state()
        return r

    def get_velocity(self) -> np.ndarray:
        _, vx = self.get_state()
        return vx

    def set_state(
        self, position: npt.ArrayLike, velocity: Optional[npt.ArrayLike] = None
    ):
        kwargs = (
            {}
            if velocity is None
            else {"targetVelocity": np.atleast_1d(velocity).tolist()}
        )
        self.bullet_client.resetJointStateMultiDof(
            bodyUniqueId=self.body_id,
            jointIndex=self.index,
            targetValue=np.atleast_1d(position).tolist(),
            **kwargs,
        )

    def _set_motor_control(
        self,
        control_mode: int,
        position: Optional[npt.ArrayLike] = None,
        velocity: Optional[npt.ArrayLike] = None,
        force: Optional[npt.ArrayLike] = None,
        position_gain: Optional[float] = None,
        velocity_gain: Optional[float] = None,
        max_velocity: Optional[float] = None,
    ):
        kwargs = {
            k: v
            for k, v in dict(
                targetPosition=None
                if position is None
                else np.array(position).reshape(-1).tolist(),
                targetVelocity=None
                if velocity is None
                else np.array(velocity).reshape(-1).tolist(),
                force=None if force is None else np.array(force).reshape(-1).tolist(),
                positionGain=position_gain,
                velocityGain=velocity_gain,
                maxVelocity=max_velocity,
            ).items()
            if v is not None
        }

        self.bullet_client.setJointMotorControlMultiDof(
            bodyUniqueId=self.body_id,
            jointIndex=self.index,
            controlMode=control_mode,
            **kwargs,
        )

    def set_position(
        self,
        position: npt.ArrayLike,
        velocity: Optional[npt.ArrayLike] = None,
        max_motor_force: Optional[npt.ArrayLike] = None,
        position_gain: Optional[float] = None,
        velocity_gain: Optional[float] = None,
        max_velocity: Optional[float] = None,
    ):
        self._set_motor_control(
            control_mode=pybullet.POSITION_CONTROL,
            position=position,
            velocity=velocity,
            force=max_motor_force,
            position_gain=position_gain,
            velocity_gain=velocity_gain,
            max_velocity=max_velocity,
        )

    def set_velocity(self, velocity: npt.ArrayLike):
        self._set_motor_control(
            control_mode=pybullet.VELOCITY_CONTROL,
            velocity=velocity,
        )

    def set_torque(self, torque: npt.ArrayLike):
        self._set_motor_control(
            control_mode=pybullet.TORQUE_CONTROL,
            force=torque,
        )

    def disable_motor(self):
        p, v, f = (0, 0, 0)
        if self.joint_type == pybullet.JOINT_SPHERICAL:
            p = np.array([0, 0, 0, 1])  # getQuaternionFromEuler([0, 0, 0])
            v = np.zeros(3)
            f = np.zeros(3)
        self.set_position(
            position=p,
            velocity=v,
            max_motor_force=f,
            position_gain=0.1,
            velocity_gain=0.1,
        )


@dataclass
class BodyPart:
    bullet_client: BulletClient
    body_id: int
    index: int
    name: str
    init_pos: np.ndarray = field(init=False)
    init_orn: np.ndarray = field(init=False)

    def __post_init__(self):
        self.init_pos = self.pos()
        self.init_orn = self.orn()

    def _link_state(self, with_velocity: bool = False) -> list[npt.ArrayLike]:
        if self.index == -1:
            return self.bullet_client.getBasePositionAndOrientation(self.body_id)
        return self.bullet_client.getLinkState(
            bodyUniqueId=self.body_id,
            linkIndex=self.index,
            computeLinkVelocity=1 if with_velocity else 0,
        )

    # def _link_state(self, with_velocity: bool = False) -> dict[str, np.ndarray]:
    #    states = self.bullet_client.getLinkState(
    #        bodyUniqueId=self.body_id,
    #        linkIndex=self.index,
    #        computeLinkVelocity=1 if with_velocity else 0,
    #    )
    #    link_state = {
    #        "COM_POSE_GLOBAL": np.concatenate([states[0], states[1]]),
    #        "INERTIAL_FRAME_LOCAL": np.concatenate([states[2], states[3]]),
    #        "LINK_FRAME_GLOBAL": np.concatenate([states[4], states[5]])
    #    }
    #    if with_velocity:
    #        link_state |= {
    #            "LINEAR_VELOCITY_GLOBAL": np.array(states[6]),
    #            "ANGULAR_VELOCITY_GLOBAL": np.array(states[7]),
    #        }
    #    return link_state

    def pose(self) -> np.ndarray:
        """Pose of center of mass in global frame"""
        state = self._link_state()
        return np.concatenate([state[0], state[1]])

    def frame_transform(self) -> np.ndarray:
        """Pose of URDF link frame in global frame"""
        state = self._link_state()
        return np.concatenate([state[4], state[5]])

    def pos(self) -> np.ndarray:
        """Position of center of mass in global frame"""
        return np.array(self._link_state()[0])

    def orn(self) -> np.ndarray:
        """Orientation of center of mass in global frame"""
        return np.array(self._link_state()[1])

    def link_pos(self) -> np.ndarray:
        """Position of URDF link frame in global frame"""
        return np.array(self._link_state()[4])

    def link_orn(self) -> np.ndarray:
        """Orientation of URDF link frame in global frame"""
        return np.array(self._link_state()[5])

    def linear_velocity(self) -> np.ndarray:
        """Linear velocity in global frame"""
        return np.array(self._link_state(with_velocity=True)[6])

    def angular_velocity(self) -> np.ndarray:
        """Angular velocity in global frame"""
        return np.array(self._link_state(with_velocity=True)[7])

    def contact_list(self):
        return self.bullet_client.getContactPoints(self.body_id, -1, self.index, -1)

    def aabb(self) -> np.ndarray:
        return np.array(self.bullet_client.getAABB(self.body_id, self.index))


@dataclass
class Body:
    bullet_client: BulletClient
    id: int
    name: str
    base_name: InitVar[str]
    parts: dict[str, BodyPart]
    joints: dict[str, Joint]
    motor_joints: list[Joint] = field(init=False)
    base: BodyPart = field(init=False)
    init_pos: np.ndarray = field(init=False)
    init_orn: np.ndarray = field(init=False)

    def __post_init__(self, base_name: str):
        self.base = self.parts[base_name]
        self.init_pos = self.pos()
        self.init_orn = self.orn()
        self.motor_joints = [
            j for j in self.joints.values() if j.joint_type != pybullet.JOINT_FIXED
        ]

    @classmethod
    def from_urdf(
        cls,
        bullet_client: BulletClient,
        file_path: Path,
        base_position: Optional[npt.ArrayLike] = None,
        base_orientation: Optional[npt.ArrayLike] = None,
        use_maximal_coordinates: Optional[bool] = None,
        fixed_base: Optional[bool] = None,
        flags: Optional[int] = None,
        global_scaling: Optional[float] = None,
    ):
        kwargs = {
            k: v
            for k, v in dict(
                basePosition=base_position,
                baseOrientation=base_orientation,
                useMaximalCoordinates=use_maximal_coordinates,
                useFixedBase=fixed_base,
                flags=flags,
                globalScaling=global_scaling,
            ).items()
            if v is not None
        }
        body_id = bullet_client.loadURDF(fileName=str(file_path.resolve()), **kwargs)
        return cls.from_id(bullet_client, body_id)

    @classmethod
    def from_id(cls, bullet_client: BulletClient, body_id: int):
        parts = {}
        joints = {}

        base_name, body_name = [
            x.decode("utf8") for x in bullet_client.getBodyInfo(body_id)
        ]

        # if bullet_client.getNumJoints(body_id) == 0:
        parts[base_name] = BodyPart(
            bullet_client=bullet_client,
            body_id=body_id,
            index=-1,
            name=base_name,
        )

        for joint_index in range(bullet_client.getNumJoints(body_id)):
            joint = Joint(bullet_client, body_id, joint_index)

            parts[joint.link_name] = BodyPart(
                bullet_client=bullet_client,
                body_id=body_id,
                index=joint_index,
                name=joint.link_name,
            )

            # if nothing else works, we take this as robot_body
            # if joint_index == 0 and robot_name not in parts.keys():
            #    parts[robot_name] = BodyPart(
            #        bullet_client=bullet_client,
            #        self.body_id=self.body_id,
            #        index=-1,
            #        name=robot_name,
            #    )

            if joint.name[:6] == "ignore":
                joint.disable_motor()
                continue

            if joint.name[:8] != "jointfix":
                joints[joint.name] = joint
                # ordered_joints.append(joint)

        return cls(
            bullet_client=bullet_client,
            id=body_id,
            name=body_name,
            base_name=base_name,
            parts=parts,
            joints=joints,
        )

    def pose(self) -> np.ndarray:
        pos, orn = self.bullet_client.getBasePositionAndOrientation(self.id)
        return np.concatenate([pos, orn])

    def pos(self) -> np.ndarray:
        return self.pose()[:3]

    def orn(self) -> np.ndarray:
        return self.pose()[3:]

    def get_part_by_index(self, index: int) -> BodyPart:
        for part in self.parts.values():
            if index == part.index:
                return part
        raise ValueError(f"Part with index {index} not found.")

    def reset_velocity(
        self,
        linearVelocity: npt.ArrayLike = [0, 0, 0],
        angularVelocity: npt.ArrayLike = [0, 0, 0],
    ):
        self.bullet_client.resetBaseVelocity(self.id, linearVelocity, angularVelocity)

    def reset_pose(self, position: npt.ArrayLike, orientation: npt.ArrayLike):
        self.bullet_client.resetBasePositionAndOrientation(
            self.id,
            position,
            orientation,
        )

    def reset_to_initial_pose(self):
        self.reset_pose(self.init_pos, self.init_orn)

    def aabb(self, part_names: Optional[list[str]] = None) -> np.ndarray:
        if part_names is None or len(part_names) == 0:
            return np.array(self.bullet_client.getAABB(self.id))
        aabb = np.stack([np.full(3, np.inf), np.full(3, -np.inf)])
        for part_name in part_names:
            part_aabb = self.parts[part_name].aabb()
            aabb[0, :] = np.minimum(aabb[0, :], part_aabb[0, :])
            aabb[1, :] = np.maximum(aabb[1, :], part_aabb[1, :])
        return aabb
