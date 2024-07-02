from mp_pytorch.mp.mp_factory import DMP
import torch
from abc import ABC, abstractmethod
from addict import Dict
from typing import List, Optional, Tuple, Union
from mp_pytorch.mp import MPFactory
from movement_primitive_diffusion.utils.matrix import unsqueeze_and_expand
from enum import Enum, unique


@unique
class TrajectoryType(Enum):
    POSITION = 0
    VELOCITY = 1


class TrajectoryHandler(ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractmethod
    def encode(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def decode(self, *args, **kwargs):
        raise NotImplementedError

    @property
    @abstractmethod
    def encoding_size(self):
        raise NotImplementedError


class ProDMPHandler(TrajectoryHandler):
    def __init__(
        self,
        num_dof: int,
        dt: float,
        traj_steps: int,
        traj_type: Union[TrajectoryType, str] = TrajectoryType.POSITION,
        weights_scale: Optional[List[float]] = None,
        precompute_dt: float = 0.01,
        num_basis: int = 10,
        tau_factor: float = 1.0,
        basis_bandwidth_factor: float = 2.0,
        alpha: float = 25.0,
        alpha_phase: float = 2.0,
        num_basis_outside: int = 0,
        goal_scale: float = 1.0,
        mp_type: str = "prodmp",
        learn_tau=False,
        learn_delay=False,
        relative_goal=False,
        auto_scale_basis=False,
        disable_goal=False,
        device: Union[str, torch.device] = "cpu",
    ):
        """ProDMPHandler, provides an interface to the ProDMP model.

        Mostly used to convert between trajectories and ProDMP parameters
        Args:
            num_dof: Number of degrees of freedom in the trajectory
            dt: Time step of the trajectory
            traj_steps: Number of steps in the trajectory
            traj_type: The type of trajectory to return in decode. POSITION or VELOCITY.
            weights_scale: Scaling of the weights
            precompute_dt: Time step for precomputation inside the ProDMP
            num_basis: Number of basis functions
            tau_factor: Factor to scale the tau parameter
            basis_bandwidth_factor: Factor to scale the bandwidth of the basis functions
            alpha: Gain factor for the forcing term
            alpha_phase: Steepness of the phase term
            num_basis_outside: Number of basis functions outside the range of the trajectory
            goal_scale: Multiplies params of ProDMP by this value
            mp_type: Type of MP to use (prodmp)
            learn_tau: Whether to learn the tau parameter
            learn_delay: Whether to learn delay
            TODO: learn_alpha_phase: Whether to learn alpha_phase
            relative_goal: Whether to use relative goal
            auto_scale_basis: Whether to automatically scale the basis functions
            disable_goal: Whether to disable the goal
            device: device on which to initialize the Motion Primitive
        """
        super().__init__()

        if dt < precompute_dt:
            raise RuntimeWarning(f"{dt=} is smaller than {precompute_dt=}. This is probably incorrect.")

        prodmp_config, num_param = self.get_prodmp_config(
            num_dof=num_dof,
            dt=dt,
            traj_steps=traj_steps,
            weights_scale=weights_scale,
            precompute_dt=precompute_dt,
            num_basis=num_basis,
            tau_factor=tau_factor,
            basis_bandwidth_factor=basis_bandwidth_factor,
            alpha=alpha,
            alpha_phase=alpha_phase,
            num_basis_outside=num_basis_outside,
            goal_scale=goal_scale,
            mp_type=mp_type,
            learn_tau=learn_tau,
            learn_delay=learn_delay,
            relative_goal=relative_goal,
            auto_scale_basis=auto_scale_basis,
            disable_goal=disable_goal,
            device=device,
        )

        # traj_time is the actual time length of the sequence that we predict.
        traj_time = dt * traj_steps

        self.prodmp_config = prodmp_config
        self.traj_steps = traj_steps
        self.traj_time = traj_time
        self.dt = dt
        self.precompute_dt = precompute_dt
        self.num_param = num_param
        self.learn_tau = learn_tau
        self.learn_delay = learn_delay
        self.learn_alpha_phase = False  # TODO

        if isinstance(traj_type, str):
            traj_type = TrajectoryType[traj_type]
        else:
            if not isinstance(traj_type, TrajectoryType):
                raise ValueError(f"Please pass traj_type either as str or TrajectoryType. Received {type(traj_type)=}.")
        self.traj_type = traj_type

        self.mp = MPFactory.init_mp(**self.prodmp_config.to_dict())

    @property
    def encoding_size(self):
        return self.num_param

    def get_prodmp_config(
        self,
        num_dof: int,
        dt: float,
        traj_steps: int,
        weights_scale: Optional[List[float]],
        precompute_dt: float = 0.01,
        num_basis: int = 10,
        tau_factor: float = 1,
        basis_bandwidth_factor: float = 2,
        alpha: float = 25,
        alpha_phase: float = 2,
        num_basis_outside: int = 0,
        goal_scale: float = 1,
        mp_type: str = "prodmp",
        learn_tau=False,
        learn_delay=False,
        relative_goal=False,
        auto_scale_basis=False,
        disable_goal=False,
        device: Union[str, torch.device] = "cpu",
    ):
        """ProDMPHandler, provides an interface to the ProDMP model.

        Mostly used to convert between trajectories and ProDMP parameters
        Args:
            num_dof: Number of degrees of freedom in the trajectory
            dt: Time step of the trajectory
            traj_steps: Number of steps in the trajectory
            weights_scale: Scaling of the weights
            precompute_dt: Time step for precomputation inside the ProDMP
            num_basis: Number of basis functions
            tau_factor: Factor to scale the tau parameter
            basis_bandwidth_factor: Factor to scale the bandwidth of the basis functions
            alpha: Gain factor for the forcing term
            alpha_phase: Steepness of the phase term
            num_basis_outside: Number of basis functions outside the range of the trajectory
            goal_scale: Multiplies params of ProDMP by this value
            mp_type: Type of MP to use (prodmp)
            learn_tau: Whether to learn the tau parameter
            learn_delay: Whether to learn delay
            relative_goal: Whether to use relative goal
            auto_scale_basis: Whether to automatically scale the basis functions
            disable_goal: Whether to disable the goal
            device: device on which to initialize the Motion Primitive
        """

        config = Dict()

        # tau is the internal time length of the ProDMP. This is generally slightly longer than traj_time
        # so that the trajectory does not end fully converged to the goal.
        tau = tau_factor * traj_steps * dt

        config.num_dof = num_dof
        config.tau = tau
        config.learn_tau = learn_tau
        config.learn_delay = learn_delay
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        config.device = device

        config.mp_type = mp_type

        config.mp_args.num_basis = num_basis
        config.mp_args.basis_bandwidth_factor = basis_bandwidth_factor
        config.mp_args.num_basis_outside = num_basis_outside
        config.mp_args.alpha = alpha
        config.mp_args.alpha_phase = alpha_phase

        # We use a smaller dt for precomputation to get smoother basis function precomputation values
        config.mp_args.dt = precompute_dt

        config.mp_args.weights_scale = weights_scale if hasattr(weights_scale, "__iter__") else torch.ones([config.mp_args.num_basis], device=device)
        config.mp_args.goal_scale = torch.tensor(goal_scale, device=device)
        config.mp_args.relative_goal = relative_goal
        config.mp_args.auto_scale_basis = auto_scale_basis
        config.mp_args.disable_goal = disable_goal

        # Generate parameters
        num_param = config.num_dof * config.mp_args.num_basis

        if not disable_goal:
            num_param += config.num_dof

        if config.learn_delay:
            num_param += 1

        if config.learn_tau:
            num_param += 1

        return config, num_param

    def encode(
        self,
        trajectories: torch.Tensor,
        times: torch.Tensor,
        initial_time: torch.Tensor,
        initial_position: torch.Tensor,
        initial_velocity: torch.Tensor,
    ) -> torch.Tensor:
        """Encodes a trajectory into ProDMP parameters

        Args:
            trajectories: Trajectories to encode [batch_size, traj_steps, num_dof]
            times: Time points for the passed trajectories [batch_size, traj_steps]
            initial_time: Time point for the initial values [batch_size]
            initial_position: Initial position [batch_size, num_dof]
            initial_velocity: Initial velocity [batch_size, num_dof]
        """

        params_dict = self.mp.learn_mp_params_from_trajs(
            times=times,
            trajs=trajectories,
            init_time=initial_time,
            init_pos=initial_position,
            init_vel=initial_velocity,
        )
        return params_dict["params"]

    def decode(
        self,
        params: torch.Tensor,
        params_L: Optional[torch.Tensor] = None,
        initial_time: Optional[torch.Tensor] = None,
        initial_position: Optional[torch.Tensor] = None,
        initial_velocity: Optional[torch.Tensor] = None,
        times: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Decodes ProDMP parameters into a trajectory

        Args:
            params: ProDMP parameters [batch_size, num_param]
            params_L: Cholesky decomposition of the parameters
            initial_time: Initial time of the trajectory
            initial_position: Initial position of the trajectory
            initial_velocity: Initial velocity of the trajectory
            times: Times at which to evaluate the trajectory
        """
        batch_size = params.size(0)
        num_param = params.size(-1)

        assert num_param == self.num_param, "Number of parameters in params does not match the number of parameters in the ProDMPHandler"

        if times is None:
            # We pass the previous action (t - dt) as the initial condition for time step t because ProDMP does not allow for non-zero initial velocities for initial_time < 0
            # We predict the trajectory for time steps [dt, 2*dt, ..., traj_time] and use them as the action sequence for time steps [0, dt, ..., traj_time - dt]
            # torch.arange does not include the end point -> [start, end) -> add dt to the end point
            times = unsqueeze_and_expand(torch.arange(self.dt, self.traj_time + self.dt, self.dt), dim=0, size=batch_size)
            # NOTE: index the first self.traj_steps to counteract possible rounding errors
            times = times[:, :self.traj_steps].to(params.device)

        initial_time = initial_time if initial_time is not None else torch.zeros(batch_size, device=params.device)
        initial_position = initial_position if initial_position is not None else torch.zeros(batch_size, self.prodmp_config.num_dof)
        initial_velocity = initial_velocity if initial_velocity is not None else torch.zeros(batch_size, self.prodmp_config.num_dof)

        self.mp.update_inputs(
            times=times,
            params=params,
            params_L=params_L,
            init_time=initial_time,
            init_pos=initial_position,
            init_vel=initial_velocity,
        )

        return_value = self.mp.get_traj_pos() if self.traj_type == TrajectoryType.POSITION else self.mp.get_traj_vel()
        if params_L is not None:
            assert not isinstance(self.mp, DMP), "Covariance not supported for non-probalistic MPs."
            uncertainty = self.mp.get_traj_pos_cov() if self.traj_type == TrajectoryType.POSITION else self.mp.get_traj_vel_cov()
            return_value = (return_value, uncertainty)

        return return_value
