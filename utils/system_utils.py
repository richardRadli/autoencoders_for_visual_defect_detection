import colorlog
import cv2
import json
import logging
import numpy as np
import os
import re
import time

from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Callable, List, Union


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


def set_img_color(img: np.ndarray, predict_mask: np.ndarray, weight_foreground: float, grayscale: bool) -> np.ndarray:
    """
    Modify image colors based on a predicted mask.

        img: Input image as a NumPy array.
        predict_mask: Predicted mask as a binary NumPy array.
        weight_foreground: Weight for blending the modified image with the original.
        grayscale:
    Returns: Modified image as a NumPy array.
    """

    if grayscale:
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


def file_reader(file_path: str, extension: str, extension2: str = None):
    """
    Reads and sorts files with specific extensions from a directory.

    Args:
        file_path (str): Path to the directory.
        extension (str): Primary file extension to search for.
        extension2 (str | None): Secondary file extension to search for if no files are found with primary.

    Returns:
        list[str]: A sorted list of file paths.
    """

    files = sorted(
        [str(file) for file in Path(file_path).glob(f"*.{extension}")],
        key=numerical_sort,
    )
    if not files and extension2:
        files = sorted(
            [str(file) for file in Path(file_path).glob(f"*.{extension2}")],
            key=numerical_sort,
        )

    return files

def resolve_num_workers(config_num_workers: int) -> int:
    """
    Resolve the number of worker processes for parallel processing.

    Args:
        config_num_workers (int): Number of workers requested by the config.

    Returns:
        int: The larger of the config value and the CPU core count (at least 1).
    """

    return max(config_num_workers, os.cpu_count() or 1)

def find_latest_file_in_latest_directory(path: str, extension: str | None = None) -> str:
    """
    Finds the latest file in the latest directory within the given path.

    Args:
        path: The path to the directory where we should look for the latest file.
        extension: Optional extension filter (e.g. ".pt"); if given, only files
            ending with it are considered.

    Returns:
        str: The path to the latest file.

    Raises:
        ValueError: When no directories or matching files are found.
    """

    dirs = [os.path.join(path, d) for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

    if not dirs:
        raise ValueError(f"No directories found in {path}")

    dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    latest_dir = dirs[0]
    files = [os.path.join(latest_dir, f) for f in os.listdir(latest_dir)
             if os.path.isfile(os.path.join(latest_dir, f))
             and (extension is None or f.endswith(extension))]

    if not files:
        raise ValueError(f"No files found in {latest_dir}")

    files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    latest_file = files[0]
    logging.info(f"The latest file is {latest_file}")

    return latest_file

def find_latest_directory(path: str) -> str:
    """
    Find the most recently modified subdirectory within the given path.

    Args:
        path (str): Directory to search for subdirectories.

    Returns:
        str: Path to the latest subdirectory.

    Raises:
        ValueError: If no subdirectories are found.
    """

    dirs = [os.path.join(path, d) for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

    if not dirs:
        raise ValueError(f"No directories found in {path}")

    dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    latest_dir = dirs[0]
    logging.info(f"The latest directory is {latest_dir}")

    return latest_dir


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

def save_list_to_json(filename: str, results_dict: dict) -> None:
    """
    Save metrics to a JSON file.

    Args:
        filename: Path to the JSON file where the lists will be saved.
        results_dict:
    Returns:
        None
    """
    
    with open(filename, "w") as json_file:
        json.dump(results_dict, json_file, indent=4)



