from pathlib import Path
from typing import List, Optional, Union
import numpy as np
from moviepy.video.fx.resize import resize
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip


def save_video_from_array(frames: List[np.ndarray], fps: Union[float, int], file_path: Union[str, Path], resize_height: Optional[int] = None) -> None:
    """Creates a video from an array of frames and saves it to a file.

    Parameters:
        frames (List[np.ndarray]): List of frames to be used in the video.
        fps (Union[float, int]): Frames per second for the output video.
        file_path (Union[str, Path]): Path to the output video file (e.g., 'output.mp4').
        resize_height (Optional[int]): If specified, the frames will be resized to this height.
    """

    # Create a VideoClip from the frames and set the duration
    video_clip = ImageSequenceClip(list(frames), fps=fps)

    if resize_height is not None:
        # Resize the video to the specified height
        video_clip = video_clip.fx(resize, height=resize_height)

    # Write the video to the specified output file
    video_clip.write_videofile(str(file_path), codec="libx264", audio=False, fps=fps)
