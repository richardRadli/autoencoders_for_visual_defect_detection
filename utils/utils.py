import colorlog
import cv2
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import time
import torch
import torch.nn as nn
import torchvision

from datetime import datetime
from functools import wraps

from utils.ssim_loss import SSIMLoss


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------- S E T U P   L O G G E R ----------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------------------------------------------------
# --------------------------------------- U S E   G P U   I F   A V A I L A B L E --------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def use_gpu_if_available() -> torch.device:
    """
    Provides information about the currently available GPUs and returns a torch device for training and inference.

    :return: A torch device for either "cuda" or "cpu".
    """

    if torch.cuda.is_available():
        cuda_info = {
            'CUDA Available': [torch.cuda.is_available()],
            'CUDA Device Count': [torch.cuda.device_count()],
            'Current CUDA Device': [torch.cuda.current_device()],
            'CUDA Device Name': [torch.cuda.get_device_name(0)]
        }

        df = pd.DataFrame(cuda_info)
        logging.info(df)
    else:
        logging.info("Only CPU is available!")

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------------- C R E A T E   T I M E S T A M P ------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def create_timestamp() -> str:
    """
    Creates a timestamp in the format of '%Y-%m-%d_%H-%M-%S', representing the current date and time.

    :return: The timestamp string.
    """

    return datetime.now().strftime('%Y-%m-%d_%H-%M-%S')


def get_patch(image, new_size, stride):
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


def patch2img(patches, im_size, patch_size, stride):
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


def fill_hole(mask):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    len_contour = len(contours)
    contour_list = []
    for i in range(len_contour):
        drawing = np.zeros_like(mask, np.uint8)  # create a black image
        img_contour = cv2.drawContours(drawing, contours, i, (255, 255, 255), -1)
        contour_list.append(img_contour)

    out = sum(contour_list)
    return out


def bg_mask(img, value, mode):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(img, value, 255, mode)
    thresh = fill_hole(thresh)
    if type(thresh) is int:
        return np.ones(img.shape)
    mask_ = np.ones(thresh.shape)
    mask_[np.where(thresh <= 127)] = 0
    return mask_


def set_img_color(img, predict_mask, weight_foreground):
    origin = img
    img[np.where(predict_mask == 255)] = (0, 0, 255)
    cv2.addWeighted(img, weight_foreground, origin, (1 - weight_foreground), 0, img)
    return img


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------- F I N D   L A T E S T   F I L E   I N   L A T E S T   D I R E C T O R Y ----------------------
# ----------------------------------------------------------------------------------------------------------------------
def find_latest_file_in_latest_directory(path: str) -> str:
    """
    Finds the latest file in the latest directory within the given path.

    :param path: str, the path to the directory where we should look for the latest file
    :return: str, the path to the latest file
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


# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------- M E A S U R E   E X E C U T I O N   T I M E ------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def measure_execution_time(func):
    """
    Decorator to measure the execution time.

    :param func:
    :return:
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        logging.info(f"Execution time of {func.__name__}: {execution_time} seconds")
        return result
    return wrapper


def visualize_images(clean_images, outputs, epoch, batch_idx, dir_path, noise_images=None):
    filename = os.path.join(dir_path, f"{epoch}_{batch_idx}.png")

    clean_images_grid = torchvision.utils.make_grid(clean_images.cpu(), nrow=8, normalize=True)
    noise_images_grid = None
    if noise_images is not None:
        noise_images_grid = torchvision.utils.make_grid(noise_images.cpu(), nrow=8, normalize=True)
    outputs_grid = torchvision.utils.make_grid(outputs.cpu(), nrow=8, normalize=True)

    plt.figure(figsize=(12, 12))

    num_of_rows = 3 if noise_images is not None else 2

    plt.subplot(num_of_rows, 1, 1)
    plt.imshow(clean_images_grid.permute(1, 2, 0))
    plt.title(f'Clean Images - Epoch {epoch}, Batch {batch_idx}')

    if noise_images is not None:
        plt.subplot(num_of_rows, 1, 2)
        plt.imshow(noise_images_grid.permute(1, 2, 0))
        plt.title(f'Noisy Images - Epoch {epoch}, Batch {batch_idx}')

    plt.subplot(num_of_rows, 1, num_of_rows)
    plt.imshow(outputs_grid.permute(1, 2, 0))
    plt.title(f'Reconstructed Images - Epoch {epoch}, Batch {batch_idx}')

    plt.savefig(filename)
    plt.close()


# ------------------------------------------------------------------------------------------------------------------
# --------------------------------------- G E T   L O S S   F U N C T I O N ----------------------------------------
# ------------------------------------------------------------------------------------------------------------------
def get_loss_function(loss_function_type):
    """

    :param loss_function_type:
    :return:
    """

    loss_functions = {
        "mse": nn.MSELoss(),
        "ssim": SSIMLoss()
    }

    if loss_function_type in loss_functions:
        return loss_functions[loss_function_type]
    else:
        raise ValueError(f"Wrong loss function type {loss_function_type}")


# ------------------------------------------------------------------------------------------------------------------
# ---------------------------------------- C R E A T E   S A V E   D I R S -----------------------------------------
# ------------------------------------------------------------------------------------------------------------------
def create_save_dirs(directory_path, network_type, timestamp):
    """

    :param directory_path:
    :param network_type:
    :param timestamp:
    :return:
    """

    directory_to_create = (
        os.path.join(directory_path, network_type, f"{timestamp}")
    )
    os.makedirs(directory_to_create, exist_ok=True)
    return directory_to_create
