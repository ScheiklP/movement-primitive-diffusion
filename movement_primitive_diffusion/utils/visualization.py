import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import matplotlib.gridspec as gridspec

from typing import Optional, Sequence


# Adapted from https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/vec_env/base_vec_env.py
def tile_images(images_nhwc: Sequence[np.ndarray]) -> np.ndarray:  # pragma: no cover
    """Tile N images into one big PxQ image

    (P,Q) are chosen to be as close as possible, and if N is square, then P=Q.

    Args:
        images_nhwc (Sequence[np.ndarray]): (n_images, height, width, n_channels)

    Returns:
        np.ndarray: img_HWc, ndim=3

    """
    img_nhwc = np.asarray(images_nhwc)
    n_images, height, width, n_channels = img_nhwc.shape
    # new_height was named H before
    new_height = int(np.ceil(np.sqrt(n_images)))
    # new_width was named W before
    new_width = int(np.ceil(float(n_images) / new_height))
    img_nhwc = np.array(list(img_nhwc) + [img_nhwc[0] * 0 for _ in range(n_images, new_height * new_width)])
    # img_HWhwc
    out_image = img_nhwc.reshape((new_height, new_width, height, width, n_channels))
    # img_HhWwc
    out_image = out_image.transpose(0, 2, 1, 3, 4)
    # img_Hh_Ww_c
    out_image = out_image.reshape((new_height * height, new_width * width, n_channels))
    return out_image


def plot_subsequence(subsequence: torch.Tensor, reference: Optional[torch.Tensor] = None, full_batch: bool = False) -> None:
    batch_size = subsequence.shape[0]
    num_dims = subsequence.shape[2]

    minibatch_index_to_plot = np.random.randint(0, batch_size)

    plt.figure(figsize=(20, 20))

    gs = gridspec.GridSpec(num_dims, 1)

    if full_batch:
        indices_to_plot = list(range(batch_size))
    else:
        indices_to_plot = [minibatch_index_to_plot]

    for batch_index in indices_to_plot:
        for i in range(num_dims):
            axs = plt.subplot(gs[i, 0])
            axs.plot(subsequence[batch_index, :, i].cpu().numpy(), label=f"Predicted_{batch_index}")
            axs.set_title(f"Dimension {i}")
            axs.set_ylabel("Value")
            axs.set_xlabel("Time Step")
            if reference is not None:
                axs.plot(reference[batch_index, :, i].cpu().numpy(), label=f"Reference_{batch_index}")
            plt.legend()

    plt.tight_layout()
    plt.show()


def plot_diffusion_steps(denoiser_outputs: torch.Tensor, sampler_states: torch.Tensor, stride: int = 1):
    diffusion_steps = denoiser_outputs.shape[0]
    num_dims = denoiser_outputs.shape[2]

    plt.figure(figsize=(20, 20))

    gs = gridspec.GridSpec(num_dims, 2)

    colors = pl.cm.jet(np.linspace(1, 0, diffusion_steps))

    for step in range(0, diffusion_steps, stride):
        for i in range(num_dims):
            ylim = (
                min(
                    denoiser_outputs[:, :, i].min().cpu().numpy(),
                    sampler_states[:, :, i].min().cpu().numpy(),
                ),
                max(
                    denoiser_outputs[:, :, i].max().cpu().numpy(),
                    sampler_states[:, :, i].max().cpu().numpy(),
                ),
            )

            axs = plt.subplot(gs[i, 0])
            axs.plot(sampler_states[step, :, i].cpu().numpy(), color=colors[step])
            axs.set_ylim(ylim)

            if i == 0:
                axs.set_title(f"Sampler")

            axs = plt.subplot(gs[i, 1])
            axs.plot(denoiser_outputs[step, :, i].cpu().numpy(), color=colors[step])
            axs.set_ylim(ylim)

            if i == 0:
                axs.set_title(f"Denoiser")

    plt.tight_layout()
    plt.show()


def plot_denoiser_step(action: torch.Tensor, denoiser_output: torch.Tensor, noised_action: torch.Tensor):
    batch_size = denoiser_output.shape[0]
    minibatch_index_to_plot = np.random.randint(0, batch_size)
    num_dims = denoiser_output.shape[2]

    plt.figure(figsize=(20, 20))

    gs = gridspec.GridSpec(num_dims, 2)

    for i in range(num_dims):
        axs = plt.subplot(gs[i, 0])
        axs.plot(action[minibatch_index_to_plot, :, i].cpu().numpy(), label=f"Action", color="green")
        axs.plot(noised_action[minibatch_index_to_plot, :, i].cpu().numpy(), label=f"Noised Action", color="red")
        axs.plot(denoiser_output[minibatch_index_to_plot, :, i].cpu().numpy(), label=f"Denoiser Output", color="blue")
        plt.legend()

        axs = plt.subplot(gs[i, 1])
        difference = action[minibatch_index_to_plot, :, i] - denoiser_output[minibatch_index_to_plot, :, i]
        axs.plot(difference.cpu().numpy(), label=f"Difference")
        plt.legend()

    plt.tight_layout()
    plt.show()
