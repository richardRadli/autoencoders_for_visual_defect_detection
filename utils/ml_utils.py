import gc
import logging
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision

from pytorch_msssim import SSIM
from typing import Optional


def device_selector(preferred_device: str) -> torch.device:
    """
    Provides information about the currently available GPUs and returns a torch device for training and inference.

    Args:
        preferred_device: A torch device for either "cuda" or "cpu".

    Returns:
        torch.device: A torch.device object representing the selected device for training and inference.
    """

    if preferred_device not in ["cuda", "cpu"]:
        logging.warning("Preferred device is not valid. Using CPU instead.")
        return torch.device("cpu")

    if preferred_device == "cuda" and torch.cuda.is_available():
        cuda_info = {
            'CUDA Available': [torch.cuda.is_available()],
            'CUDA Device Count': [torch.cuda.device_count()],
            'Current CUDA Device': [torch.cuda.current_device()],
            'CUDA Device Name': [torch.cuda.get_device_name(0)]
        }

        df = pd.DataFrame(cuda_info)
        logging.info(df)
        return torch.device("cuda")

    if preferred_device in ["cuda"] and not torch.cuda.is_available():
        logging.info("Only CPU is available!")
        return torch.device("cpu")

    if preferred_device == "cpu":
        logging.info("Selected CPU device")
        return torch.device("cpu")


def visualize_images(
    clean_images: torch.Tensor, outputs: torch.Tensor, epoch: int, batch_idx: int, dir_path: str,
        noise_images: Optional[torch.Tensor] = None) -> None:
    """
    Visualize and save images for inspection.

        clean_images: Tensor containing clean images.
        outputs: Tensor containing reconstructed images.
        epoch: Current epoch number.
        batch_idx: Current batch index.
        dir_path: Directory path to save the visualization.
        noise_images: Optional tensor containing noisy images.
    Returns: None
    """

    filename = os.path.join(dir_path, f"{epoch}_{batch_idx}.png")

    clean_images_grid = torchvision.utils.make_grid(clean_images.cpu(), nrow=8, normalize=True)
    noise_images_grid = None
    if noise_images is not None:
        noise_images_grid = torchvision.utils.make_grid(noise_images.cpu(), nrow=8, normalize=True)
    outputs_grid = torchvision.utils.make_grid(outputs.cpu(), nrow=8, normalize=True)

    plt.figure(figsize=(15, 5))  # Adjust the figsize to fit horizontally

    num_of_plots = 2 if noise_images is None else 3

    plt.subplot(1, num_of_plots, 1)
    plt.imshow(clean_images_grid.permute(1, 2, 0))
    plt.title(f'Clean Images - Epoch {epoch}, Batch {batch_idx}')
    plt.axis('off')  # Optional: hide the axes for better visualization

    if noise_images is not None:
        plt.subplot(1, num_of_plots, 2)
        plt.imshow(noise_images_grid.permute(1, 2, 0))
        plt.title(f'Noisy Images - Epoch {epoch}, Batch {batch_idx}')
        plt.axis('off')  # Optional: hide the axes for better visualization

    plt.subplot(1, num_of_plots, num_of_plots)
    plt.imshow(outputs_grid.permute(1, 2, 0))
    plt.title(f'Reconstructed Images - Epoch {epoch}, Batch {batch_idx}')
    plt.axis('off')  # Optional: hide the axes for better visualization

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.savefig(filename, dpi=300)
    plt.close()
    gc.collect()



def get_loss_function(loss_function_type: str, grayscale=None):
    """
    Get the loss function based on the provided loss function type.

    Args:
        loss_function_type: String specifying the type of loss function ("mse" or "ssim").
        grayscale: Optional boolean specifying whether to use grayscale or not.

    Returns:
         Loss function instance.
    """

    loss_functions = {
        "mse": nn.MSELoss(),
        "ssim": SSIM(win_sigma=1.5, data_range=1, size_average=True, channel=1 if grayscale else 3),
    }

    if loss_function_type in loss_functions:
        return loss_functions[loss_function_type]
    else:
        raise ValueError(f"Wrong loss function type {loss_function_type}")

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # If running on the CuDNN backend
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)

def patch2img(patches, im_size: int, patch_size: int, stride: int) -> np.ndarray:
    """
    Reconstruct a full image from overlapping patches with seamless blending.

    Each patch is weighted by a smooth 2D window so overlapping regions blend
    gradually instead of showing hard seams. The weighted patches are summed
    with fold and divided by the summed weights.

    Args:
        patches: Model output patches, tensor of shape (num_patches, C, patch_size, patch_size).
        im_size: Size of the reconstructed square image.
        patch_size: Size of each square patch.
        stride: Stride between consecutive patches.

    Returns:
        np.ndarray: The reconstructed image of shape (im_size, im_size, C).
    """
    import torch
    from torch.nn.functional import fold

    patches = patches.detach().cpu()
    num_patches, channels, _, _ = patches.shape

    window_1d = torch.hann_window(patch_size, periodic=False)
    window_2d = 0.1 + 0.9 * torch.outer(window_1d, window_1d)

    weighted = (patches * window_2d).permute(1, 2, 3, 0).reshape(
        1, channels * patch_size * patch_size, num_patches
    )
    numerator = fold(weighted, output_size=(im_size, im_size), kernel_size=patch_size, stride=stride)

    weight_stack = window_2d.reshape(1, patch_size * patch_size, 1).expand(-1, -1, num_patches)
    denominator = fold(weight_stack, output_size=(im_size, im_size), kernel_size=patch_size, stride=stride)

    output = numerator / denominator.clamp(min=1e-8)
    return output.squeeze(0).permute(1, 2, 0).numpy()
