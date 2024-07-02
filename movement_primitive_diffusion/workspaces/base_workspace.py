import wandb
import hydra
import numpy as np
import time
import torch
import cv2

from pathlib import Path
from omegaconf import DictConfig
from typing import Dict, List
from tqdm import tqdm

from movement_primitive_diffusion.agents.base_agent import BaseAgent
from movement_primitive_diffusion.utils.video import save_video_from_array


class BaseWorkspace:
    def __init__(
        self,
        env_config: DictConfig,
        t_act: int,
        num_upload_successful_videos: int = 5,
        num_upload_failed_videos: int = 5,
        show_images: bool = False,
    ):
        super().__init__()

        self.env = hydra.utils.instantiate(env_config)
        self.t_act = t_act
        self.num_upload_successful_videos = num_upload_successful_videos
        self.num_upload_failed_videos = num_upload_failed_videos
        self.show_images = show_images
        self.time_limit = env_config.time_limit

    def test_agent(self, agent: BaseAgent, num_trajectories: int = 10) -> dict:
        self.num_successful_trajectories = 0
        self.num_failed_trajectories = 0
        frames_of_successful_trajectories = []
        frames_of_failed_trajectories = []

        # TODO: nicer progressbar with steps_to_go like in BaseVectorWorkspace
        for i in (pbar := tqdm(range(num_trajectories), desc="Testing agent", leave=False)):
            self.reset_env(caller_locals=locals())

            done = False
            successful = False
            episode_frames = []
            image_shape = self.render_function(caller_locals=locals()).shape

            while not done:
                observation_buffer = self.env.get_observation_dict()

                # Create torch tensors from numpy arrays
                for key, val in observation_buffer.items():
                    observation_buffer[key] = torch.from_numpy(val)

                # Process the observation buffer to get observations and extra inputs
                observation, extra_inputs = agent.process_batch.process_env_observation(observation_buffer)

                # Move observations and extra inputs to device
                for key, val in observation.items():
                    observation[key] = val.to(agent.device)
                for key, val in extra_inputs.items():
                    if isinstance(val, torch.Tensor):
                        extra_inputs[key] = val.to(agent.device)

                # Predict the next action sequence
                actions = agent.predict(observation, extra_inputs)

                # Remove batch dimension and move to cpu and get numpy array
                actions = actions.squeeze(0).cpu().numpy()
                assert actions.ndim == 2, f"Actions should be of shape (T, N) for T timesteps and N action dimensions. Got shape {actions.shape}"

                # Execute up to t_act actions in the environment
                for action in actions[: self.t_act]:
                    # Take action in environment
                    env_obs, reward, terminated, truncated, info = self.env.step(action)

                    # Render environment
                    rgb_frame = self.render_function(caller_locals=locals())
                    episode_frames.append(rgb_frame)

                    if self.show_images:
                        cv2.imshow("Workspace", rgb_frame[..., ::-1])
                        cv2.waitKey(1)

                    successful = self.check_success_hook(caller_locals=locals())

                    # Check if episode is done
                    done = truncated or terminated

                    self.post_step_hook(caller_locals=locals())

                    # End early if episode is done
                    if done:
                        break

            # Add frames of episode to buffer of successful or failed trajectories
            if successful:
                if self.num_successful_trajectories < self.num_upload_successful_videos:
                    frames_of_successful_trajectories.extend(episode_frames)
                self.num_successful_trajectories += 1
            else:
                if self.num_failed_trajectories < self.num_upload_failed_videos:
                    frames_of_failed_trajectories.extend(episode_frames)
                self.num_failed_trajectories += 1

            pbar.set_postfix(success_rate=self.num_successful_trajectories / (self.num_successful_trajectories + self.num_failed_trajectories))

            self.post_episode_hook(caller_locals=locals())

        # Log at least one black frame to have info for this epoch
        fps = int(1 / self.env.dt)
        if len(frames_of_successful_trajectories) == 0:
            frames_of_successful_trajectories.append(np.zeros(image_shape, dtype=np.uint8))
        if len(frames_of_failed_trajectories) == 0:
            frames_of_failed_trajectories.append(np.zeros(image_shape, dtype=np.uint8))
        self.log_video(frames_of_successful_trajectories, fps=fps, metric="successful")
        self.log_video(frames_of_failed_trajectories, fps=fps, metric="failed")

        return self.get_result_dict(caller_locals=locals())

    def check_success_hook(self, caller_locals: Dict) -> bool:
        """Function to modify success check behavior in subclasses.

        For example for checking if the agent reached the goal.

        """
        return caller_locals["terminated"]

    def reset_env(self, caller_locals: Dict) -> np.ndarray:
        """Function to modify reset behavior in subclasses.

        For example for setting a random seed, or passing an options dict.

        """
        return self.env.reset()

    def render_function(self, caller_locals: Dict) -> np.ndarray:
        """Function to modify render behavior in subclasses.

        For example for setting a random seed, or passing an options dict.

        """
        return self.env._render_frame(mode="rgb_array")

    def post_step_hook(self, caller_locals: Dict) -> None:
        """Function to modify post step behavior in subclasses.

        Updating the current best value of some metric.
        """
        pass

    def post_episode_hook(self, caller_locals: Dict) -> None:
        """Function to modify post episode behavior in subclasses.

        For example adding information to the progress bar.
        """
        pass

    def get_result_dict(self, caller_locals: Dict) -> Dict[str, float]:
        """Function to modify result dict in subclasses.

        For example adding information to the progress bar.
        """

        return {"success_rate": self.num_successful_trajectories / (self.num_successful_trajectories + self.num_failed_trajectories)}

    def get_result_dict_keys(self) -> List[str]:
        return ["success_rate"]

    def log_video(self, frame_buffer: List[np.ndarray], fps: int, metric: str = "video") -> None:
        # Write video with opencv
        now = time.strftime("%Y%m%d-%H%M%S")
        video_dir = Path(wandb.run.dir) / "media/videos"
        video_dir.mkdir(parents=True, exist_ok=True)
        file_path = str(((video_dir / f"{metric}_{now}.mp4").absolute()))

        save_video_from_array(frames=frame_buffer, fps=fps, file_path=file_path, resize_height=200)

        # Upload video to wandb
        wandb.log({metric: wandb.Video(file_path, fps=fps, format="mp4")})

    def close(self) -> None:
        self.env.close()
