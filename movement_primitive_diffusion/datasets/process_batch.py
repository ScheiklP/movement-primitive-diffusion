import torch
import hydra

from typing import Dict, List, Tuple, Union, Optional
from omegaconf import DictConfig

from movement_primitive_diffusion.utils.matrix import unsqueeze_and_expand
from movement_primitive_diffusion.utils.mp_utils import ProDMPHandler


class ProcessBatch:
    def __init__(
        self,
        t_obs: int,
        t_pred: int,
        action_keys: DictConfig,
        observation_keys: Union[str, List[str]],
        relative_action_values: bool = False,
        predict_past: bool = False,
    ):
        self.action_keys = []
        self.action_sizes = []
        for info in action_keys:
            self.action_keys.append(info.key)
            assert len(info.feature_size) == 1, f"Feature size of action {info.key} is not a list of length 1"
            self.action_sizes += info.feature_size

        if isinstance(observation_keys, str):
            observation_keys = [observation_keys]
        self.observation_keys = observation_keys

        self.t_obs = t_obs
        self.t_pred = t_pred
        self.predict_past = predict_past

        # Whether the action values of the sequence are relative to the first action
        self.relative_action_values = relative_action_values
        if self.relative_action_values:
            assert self.t_obs > 1, "Relative action values require at least two observations for correct indexing of the absolute start value."

    def __call__(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        # indices: [0           , ..., t_obs-3, t_obs-2, t_obs-1, t_obs, t_obs+1, ..., t_obs+t_pred-1]
        # times:   [ t-(t_obs-1), ..., t-2,     t-1,     t0,      t1,    t2,      ..., t(t_pred-1) ]
        # obs:     [obs,        , ..., obs,     obs,     obs]
        # act:                                          [act,     act,   act,     ..., act]
        # with predict_past = True:
        # act:     [act,        , ..., act,     act,    act,      act,   act,     ..., act]

        if self.predict_past:
            action = torch.cat([batch[key] for key in self.action_keys], dim=-1)
        else:
            action = torch.cat([batch[key][:, self.t_obs - 1 :] for key in self.action_keys], dim=-1)

        assert action.ndim == 3, f"Action has wrong number of dimensions: {action.ndim}"
        assert action.shape[1:] == self.action_shape, f"Action shape {action.shape[1:]} does not match expected shape {self.action_shape}"

        if self.relative_action_values:
            # Subtract the action value of time_step t-1 from the action value of time_step t0, t1, ...
            # to get relative action values.
            absolute_start = torch.cat([batch[key][:, self.t_obs - 2] for key in self.action_keys], dim=-1)
            action -= absolute_start.unsqueeze(1)

        observation = {key: batch[key][:, : self.t_obs] for key in self.observation_keys}

        extra_inputs = {}

        return action, observation, extra_inputs

    def process_env_observation(
        self,
        observation: Dict[str, torch.Tensor],
        skip_initial_values: bool = True,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        # Tensors in observation are expected to have shape (B, t_obs, ...)

        # Make sure the tensors have at least B and t_obs
        if not all([observation[key].ndim >= 2 for key in self.observation_keys]):
            dims = {key: observation[key].ndim for key in self.observation_keys}
            raise ValueError(f"Observation should have at least 2 dimensions for batch size and t_obs. Got {dims}")

        # Make sure t_obs is consistent for all tensors
        if not all([observation[key].shape[1] == self.t_obs for key in self.observation_keys]):
            lens = {key: observation[key].shape[1] for key in self.observation_keys}
            raise ValueError(f"Observation has wrong number of time steps. Expected {self.t_obs}, got {lens}")

        # Filter out the keys that are not needed
        processed_observation = {key: observation[key] for key in self.observation_keys}

        extra_inputs = {}

        return processed_observation, extra_inputs

    @property
    def action_size(self) -> int:
        return sum(self.action_sizes)

    @property
    def action_shape(self) -> Tuple[int, int]:
        if self.predict_past:
            return (self.t_obs - 1 + self.t_pred, sum(self.action_sizes))
        return (self.t_pred, sum(self.action_sizes))


class ProcessBatchProDMP(ProcessBatch):
    """

    Notes:
        **initial_values_come_from_action_data**:
        If the initial values (initial_position and initial_velocity) are taken from actual trajectory values, t0 (index t_obs - 1) holds the correct value.
        However, if they are taken from action data (desired trajectory values for the next time step), the correct time step is t-1 (index t_obs - 2).
    """

    def __init__(
        self,
        t_obs: int,
        t_pred: int,
        action_keys: DictConfig,
        observation_keys: Union[str, List[str]],
        initial_position_keys: Union[str, List[str]],
        initial_velocity_keys: Union[str, List[str]],
        initial_values_come_from_action_data: bool = False,
        relative_action_values: bool = False,
        predict_past: bool = False,
    ):

        if predict_past:
            raise NotImplementedError("predict_past is not yet implemented for the ProDMP case.")

        super().__init__(
            t_obs=t_obs,
            t_pred=t_pred,
            action_keys=action_keys,
            observation_keys=observation_keys,
            relative_action_values=relative_action_values,
            predict_past=predict_past,
        )

        if isinstance(initial_position_keys, str):
            initial_position_keys = [initial_position_keys]
        self.initial_position_keys = initial_position_keys

        if isinstance(initial_velocity_keys, str):
            initial_velocity_keys = [initial_velocity_keys]
        self.initial_velocity_keys = initial_velocity_keys

        self.initial_values_come_from_action_data = initial_values_come_from_action_data
        self.initial_value_index = self.t_obs - 2 if self.initial_values_come_from_action_data else self.t_obs - 1
        if self.initial_values_come_from_action_data:
            assert self.t_obs > 1, "If initial values come from action data, at least two observations  are required for correct indexing of the initial values."

    def __call__(
        self,
        batch: Dict[str, torch.Tensor],
    ):
        action, observation, extra_inputs = super().__call__(batch)

        # If initial_values_come_from_action_data, we assume that the initial values from the prodmp come from an array that contains action values.
        # Action values are "desired trajectory values for the **next** time step"
        # In this case, the initial values are taken from the action data of time step t-1 (index t_obs - 2).
        # Otherwise, the initial values are taken from the actual trajectory data of time step t0 (index t_obs - 1).

        # indices: [0           , ..., t_obs-3, t_obs-2, t_obs-1, t_obs, t_obs+1, ..., t_obs+t_pred-1]
        # times:   [ t-(t_obs-1), ..., t-2,     t-1,     t0,      t1,    t2,      ..., t(t_pred-1) ]
        # obs:     [obs,        , ..., obs,     obs,     obs]
        # act:                                          [act,     act,   act,     ..., act]
        #                                                ^--- initial_values_come_from_action_data==False -> initial values come from state data
        #                                       ^--- initial_values_come_from_action_data==True -> initial values come from action data

        # TODO: implement logic for predict_past. Add time values to extra_inputs -> then go to prodmp_causal_transformer_inner_model: trajectory = self.prodmp_handler.decode(params, **extra_inputs)

        extra_inputs = {
            "initial_position": torch.cat([batch[key][:, self.initial_value_index] for key in self.initial_position_keys], dim=-1),
            "initial_velocity": torch.cat([batch[key][:, self.initial_value_index] for key in self.initial_velocity_keys], dim=-1),
            **extra_inputs,
        }

        if self.relative_action_values:
            # Set initial positions to zero
            extra_inputs["initial_position"] = torch.zeros_like(extra_inputs["initial_position"])

        return action, observation, extra_inputs

    def process_env_observation(
        self,
        observation: Dict[str, torch.Tensor],
        skip_initial_values: bool = False,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        processed_observation, extra_inputs = super().process_env_observation(observation)

        if not skip_initial_values:
            # For the complete batch, get the values for the last time step
            # NOTE: Make sure that the workspace correctly sets the values for initial_<position/velocity>_keys
            # both _state and _state_action should be the valid initial condition, when setting them for t=0
            extra_inputs = {
                "initial_position": torch.cat([observation[key][:, -1] for key in self.initial_position_keys], dim=-1),
                "initial_velocity": torch.cat([observation[key][:, -1] for key in self.initial_velocity_keys], dim=-1),
            }

            if self.relative_action_values:
                # Set initial positions to zero
                extra_inputs["initial_position"] = torch.zeros_like(extra_inputs["initial_position"])

        return processed_observation, extra_inputs


class ProcessBatchProDMPParameterSpace(ProcessBatchProDMP):
    def __init__(
        self,
        t_obs: int,
        t_pred: int,
        action_keys: DictConfig,
        observation_keys: Union[str, List[str]],
        initial_position_keys: Union[str, List[str]],
        initial_velocity_keys: Union[str, List[str]],
        prodmp_handler_config: DictConfig,
        initial_values_come_from_action_data: bool = False,
        relative_action_values: bool = False,
        predict_past: bool = False,
    ):
        super().__init__(
            t_obs=t_obs,
            t_pred=t_pred,
            action_keys=action_keys,
            observation_keys=observation_keys,
            initial_position_keys=initial_position_keys,
            initial_velocity_keys=initial_velocity_keys,
            initial_values_come_from_action_data=initial_values_come_from_action_data,
            relative_action_values=relative_action_values,
            predict_past=predict_past,
        )

        prodmp_handler_config.num_dof = sum(self.action_sizes)
        self.prodmp_handler: ProDMPHandler = hydra.utils.instantiate(prodmp_handler_config)

    def __call__(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        action, observation, extra_inputs = super().__call__(batch)

        times = unsqueeze_and_expand(
            torch.arange(
                self.prodmp_handler.dt,
                self.prodmp_handler.traj_time + self.prodmp_handler.dt,
                self.prodmp_handler.dt,
            ),
            dim=0,
            size=action.size(0),
        )
        times = times[:, : self.prodmp_handler.traj_steps].to(action.device)
        parameters = self.prodmp_handler.encode(
            trajectories=action,
            times=times,
            initial_position=extra_inputs["initial_position"],
            initial_velocity=extra_inputs["initial_velocity"],
            initial_time=torch.zeros_like(extra_inputs["initial_position"][:, 0]),
        )
        extra_inputs = {
            "action": action,
            **extra_inputs,
        }

        return parameters, observation, extra_inputs

    def parameters_to_action(
        self,
        parameters: torch.Tensor,
        extra_inputs: Dict[str, torch.Tensor],
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        return self.prodmp_handler.decode(
            params=parameters,
            initial_position=extra_inputs["initial_position"],
            initial_velocity=extra_inputs["initial_velocity"],
        )

    @property
    def action_size(self) -> int:
        return self.prodmp_handler.encoding_size

    @property
    def parameter_shape(self) -> Tuple[int]:
        return (self.prodmp_handler.encoding_size,)


class ProcessBatchProDMPNLL(ProcessBatchProDMP):
    def __init__(
        self,
        t_obs: int,
        t_pred: int,
        dt: float,
        action_keys: DictConfig,
        observation_keys: Union[str, List[str]],
        initial_position_keys: Union[str, List[str]],
        initial_velocity_keys: Union[str, List[str]],
        initial_values_come_from_action_data: bool = False,
        relative_action_values: bool = False,
        additional_decoder_input_keys: Optional[List[str]] = None,
        predict_past: bool = False,
    ):
        if t_pred < 2:
            raise ValueError(f"For NLL training of ProDMPs, t_pred needs to be at least 2, as the model predicts values of pairs of time indices. {t_pred=}")

        super().__init__(
            t_obs=t_obs,
            t_pred=t_pred,
            action_keys=action_keys,
            observation_keys=observation_keys,
            initial_position_keys=initial_position_keys,
            initial_velocity_keys=initial_velocity_keys,
            initial_values_come_from_action_data=initial_values_come_from_action_data,
            relative_action_values=relative_action_values,
            predict_past=predict_past,
        )

        self.dt = dt
        self.additional_decoder_input_keys = additional_decoder_input_keys if additional_decoder_input_keys is not None else []

    def __call__(
        self,
        batch: Dict[str, torch.Tensor],
    ):
        action, observation, extra_inputs = super().__call__(batch)

        # [int, int, int, ...] (t_pred,)
        prediction_indices = torch.arange(0, self.t_pred)

        # From the prediction indices, create all possible pair combinations
        # [(int, int), (int, int), ...] (prediction_length!/((prediction_length-2)!*2!), 2) -> (combinations, 2)
        prediction_index_pairs = torch.combinations(prediction_indices, 2)

        # What are the relative timef of the prediction pairs?
        # [(float, float), (float, float), ...] (combinations, 2) e.g. [[0.1, 0.2], [0.1, 0.3], [0.2, 0.3]]
        relative_prediction_pair_times = (prediction_index_pairs + 1) * self.dt
        relative_prediction_pair_times = unsqueeze_and_expand(relative_prediction_pair_times, dim=0, size=action.shape[0])
        extra_inputs["prediction_times"] = relative_prediction_pair_times.to(action.device)

        # Set the initial time and expand all initial value tensors to shape (B, combinations, ...)
        extra_inputs["initial_time"] = torch.zeros(relative_prediction_pair_times.shape[:2], device=action.device)
        extra_inputs["initial_position"] = unsqueeze_and_expand(extra_inputs["initial_position"], dim=1, size=relative_prediction_pair_times.shape[1])
        extra_inputs["initial_velocity"] = unsqueeze_and_expand(extra_inputs["initial_velocity"], dim=1, size=relative_prediction_pair_times.shape[1])

        action = action[:, prediction_index_pairs]
        # change the axis such that value shape and pairs are switched (combinations, 2, value shape) -> swith time and value axis (combinations, value shape, 2)
        action = action.swapaxes(-1, -2)
        # change the shape from (combinations, value shape, 2) to (combinations, 2 * value shape)
        # -> (dof1 t1, dof1 t2, dof2 t1, dof2 t2, ..., dofN t1, dofN t2)
        action = action.reshape(*action.shape[:-2], -1)

        # Add any keys that should be passed to the decoder directly (bypasses the encoder)
        additional_decoder_inputs = [batch[key][:, : self.t_obs] for key in self.additional_decoder_input_keys]
        if len(additional_decoder_inputs):
            additional_decoder_inputs = torch.cat(additional_decoder_inputs, dim=-1)
        else:
            additional_decoder_inputs = None
        extra_inputs["additional_decoder_inputs"] = additional_decoder_inputs

        return action, observation, extra_inputs

    def process_env_observation(
        self,
        observation: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        processed_observation, extra_inputs = super().process_env_observation(observation)
        batch_size = extra_inputs["initial_position"].shape[0]

        # Add timepoints for trajectory prediction
        # [int, int, int, ...] (t_pred,)
        prediction_indices = torch.arange(0, self.t_pred)
        # [dt, 2*dt, 3*dt, ...]
        prediction_times = (prediction_indices + 1) * self.dt
        # Extend to batch size
        prediction_times = unsqueeze_and_expand(prediction_times, dim=0, size=batch_size)

        # Add any keys that should be passed to the decoder directly (bypasses the encoder)
        additional_decoder_inputs = [observation[key][:, : self.t_obs] for key in self.additional_decoder_input_keys]
        if len(additional_decoder_inputs):
            additional_decoder_inputs = torch.cat(additional_decoder_inputs, dim=-1)
        else:
            additional_decoder_inputs = None

        extra_inputs = {
            "initial_time": torch.zeros(batch_size),
            "prediction_times": prediction_times,
            "additional_decoder_inputs": additional_decoder_inputs,
            **extra_inputs,
        }

        return processed_observation, extra_inputs