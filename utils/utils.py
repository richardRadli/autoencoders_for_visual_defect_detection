import colorlog
import cv2
import gc
import json
import jsonschema
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import time
import torch
import torch.nn as nn
import torchvision

from datetime import datetime
from functools import wraps
from jsonschema import validate
from pathlib import Path
from pytorch_msssim import SSIM
from typing import Any, Callable, Optional, List, Union


def setup_logger():
    """
    Set up a colorized logger with the following log levels and colors:

    - DEBUG: Cyan
    - INFO: Green
    - WARNING: Yellow
    - ERROR: Red
    - CRITICAL: Red on a white background

    Returns:
        The configured logger instance.
    """

    # Check if logger has already been set up
    logger = logging.getLogger()
    if logger.hasHandlers():
        return logger

    # Set up logging
    logger.setLevel(logging.INFO)

    # Create a colorized formatter
    formatter = colorlog.ColoredFormatter(
        "%(log_color)s%(levelname)-8s%(reset)s %(white)s%(message)s",
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        })

    # Create a console handler and add the formatter to it
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


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


def create_timestamp() -> str:
    """
    Creates a timestamp in the format of '%Y-%m-%d_%H-%M-%S', representing the current date and time.

    Returns: The timestamp string.
    """

    return datetime.now().strftime('%Y-%m-%d_%H-%M-%S')


def get_patch(image: np.ndarray, new_size: int, stride: int) -> np.ndarray:
    """
    Extract square patches from an image with a specified size and stride.

        image: The input image as a NumPy array.
        new_size: The size of the patches to extract.
        stride: The stride between consecutive patches.
    Returns: An array containing extracted patches.
    """

    h, w = image.shape[:2]
    i, j = new_size, new_size
    patch = []
    while i <= h:
        while j <= w:
            patch.append(image[i - new_size:i, j - new_size:j])
            j += stride
        j = new_size
        i += stride
    return np.array(patch)


def patch2img(patches, im_size: int, patch_size: int, stride: int) -> np.ndarray:
    """
    Reconstruct an image from patches with a specified size and stride.

        patches: Patches to reconstruct, assumed to be a NumPy array or PyTorch tensor.
        im_size: Size of the reconstructed image.
        patch_size: Size of the square patches used during extraction.
        stride: The stride between consecutive patches during extraction.
    Returns: Reconstructed image.
    """

    patches = patches.detach().cpu().numpy()
    patches = np.transpose(patches, (0, 2, 3, 1))
    img = np.zeros((im_size, im_size, patches.shape[3] + 1))
    i, j = patch_size, patch_size
    k = 0
    while i <= im_size:
        while j <= im_size:
            img[i - patch_size:i, j - patch_size:j, :-1] += patches[k]
            img[i - patch_size:i, j - patch_size:j, -1] += np.ones((patch_size, patch_size))
            k += 1
            j += stride
        j = patch_size
        i += stride
    mask = np.repeat(img[:, :, -1][..., np.newaxis], patches.shape[3], 2)
    img = img[:, :, :-1] / mask
    return img


def set_img_color(img: np.ndarray, predict_mask: np.ndarray, weight_foreground: float) -> np.ndarray:
    """
    Modify image colors based on a predicted mask.

        img: Input image as a NumPy array.
        predict_mask: Predicted mask as a binary NumPy array.
        weight_foreground: Weight for blending the modified image with the original.
    Returns: Modified image as a NumPy array.
    """

    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    origin = img
    img[np.where(predict_mask == 255)] = (0, 0, 255)
    cv2.addWeighted(img, weight_foreground, origin, (1 - weight_foreground), 0, img)
    return img


def numerical_sort(value: str) -> List[Union[str, int]]:
    """
    Sorts numerical values in a string ensuring correct numerical sorting.

    Args:
        value (str): The input string containing numerical and non-numerical parts.

    Returns:
        List[Union[str, int]]: A list containing both strings and integers sorted by numerical value.
    """

    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


def file_reader(file_path: str, extension: str) -> List[str]:
    """
    Reads files with a specific extension from a given directory and sorts them numerically.

    Args:
        file_path (str): The path to the directory containing the files.
        extension (str): The extension of the files to be read.

    Returns:
        List[str]: A sorted list of filenames with the specified extension.
    """

    return sorted([str(file) for file in Path(file_path).glob(f'*.{extension}')], key=numerical_sort)


def find_latest_file_in_latest_directory(path: str) -> str:
    """
    Finds the latest file in the latest directory within the given path.

        path: str, the path to the directory where we should look for the latest file
    Returns: str, the path to the latest file
    :raise: when no directories or files found
    """

    dirs = [os.path.join(path, d) for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

    if not dirs:
        raise ValueError(f"No directories found in {path}")

    dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    latest_dir = dirs[0]
    files = [os.path.join(latest_dir, f) for f in os.listdir(latest_dir) if
             os.path.isfile(os.path.join(latest_dir, f))]

    if not files:
        raise ValueError(f"No files found in {latest_dir}")

    files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    latest_file = files[0]
    logging.info(f"The latest file is {latest_file}")

    return latest_file


def measure_execution_time(func: Callable) -> Callable:
    """
    Decorator to measure the execution time of a function.

    Args:
        func (Callable): The function to be decorated.

    Returns:
        Callable: The decorated function.
    """

    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        """
        Wrapper function to measure execution time.

        Args:
            *args: Positional arguments passed to the function.
            **kwargs: Keyword arguments passed to the function.

        Returns:
            Any: The result of the function.
        """

        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        wrapper.execution_time = end_time - start_time
        logging.info(f"Execution time of {func.__name__}: {wrapper.execution_time:.4f} seconds")
        return result

    wrapper.execution_time = None
    return wrapper


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


def get_loss_function(loss_function_type: str):
    """
    Get the loss function based on the provided loss function type.

    Args:
        loss_function_type: String specifying the type of loss function ("mse" or "ssim").

    Returns:
         Loss function instance.
    """

    loss_functions = {
        "mse": nn.MSELoss(),
        "ssim": SSIM(win_sigma=1.5, data_range=1, size_average=True, channel=1)
    }

    if loss_function_type in loss_functions:
        return loss_functions[loss_function_type]
    else:
        raise ValueError(f"Wrong loss function type {loss_function_type}")


def create_save_dirs(directory_path: str, network_type: str, timestamp: str) -> str:
    """
    Create and return a directory path based on input parameters.

    Args:
        directory_path: Base directory path where the new directory will be created.
        network_type: String specifying the type of network.
        timestamp: String timestamp for uniqueness.

    Returns:
         Created directory path.
    """

    directory_to_create = (
        os.path.join(directory_path, network_type, f"{timestamp}")
    )
    os.makedirs(directory_to_create, exist_ok=True)
    return directory_to_create


def avg_of_list(my_list):
    """
    Calculate and return the average value of the elements in the input list.

    Args:
        my_list: List of numerical values.

    Returns:
        Average value of the elements in the list.
    """

    return sum(my_list) / len(my_list)


def load_config_json(json_schema_filename: str, json_filename: str):
    """
    Args:
        json_schema_filename:
        json_filename:

    Returns:

    """

    with open(json_schema_filename, "r") as schema_file:
        schema = json.load(schema_file)

    with open(json_filename, "r") as config_file:
        config = json.load(config_file)

    try:
        validate(config, schema)
        logging.info("JSON data is valid.")
        return config
    except jsonschema.exceptions.ValidationError as err:
        logging.error(f"JSON data is invalid: {err}")


def save_list_to_json(filename: str, results_dict: dict) -> None:
    """
    Save metrics to a json file.

    Args:
        filename: Path to the json file where the lists will be saved.
        results_dict:
    Returns:
        None
    """
    
    with open(filename, "w") as json_file:
        json.dump(results_dict, json_file, indent=4)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # If running on the CuDNN backend
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
