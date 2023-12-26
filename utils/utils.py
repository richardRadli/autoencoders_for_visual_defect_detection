import colorlog
import cv2
import logging
import numpy as np
import os
import random
import pandas as pd
import torch

from datetime import datetime


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


def random_crop(image, new_size):
    h, w = image.shape[:2]
    y = np.random.randint(0, h - new_size[0])
    x = np.random.randint(0, w - new_size[1])
    image = image[y:y + new_size[0], x:x + new_size[1]]
    return image


def rotate_image(img, angle, crop):
    h, w = img.shape[:2]
    angle %= 360
    m_rotate = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    img_rotated = cv2.warpAffine(img, m_rotate, (w, h))
    if crop:
        angle_crop = angle % 180
        if angle_crop > 90:
            angle_crop = 180 - angle_crop
        theta = angle_crop * np.pi / 180.0
        hw_ratio = float(h) / float(w)
        tan_theta = np.tan(theta)
        numerator = np.cos(theta) + np.sin(theta) * tan_theta
        r = hw_ratio if h > w else 1 / hw_ratio
        denominator = r * tan_theta + 1
        crop_mult = numerator / denominator
        w_crop = int(round(crop_mult * w))
        h_crop = int(round(crop_mult * h))
        x0 = int((w - w_crop) / 2)
        y0 = int((h - h_crop) / 2)
        img_rotated = img_rotated[y0:y0 + h_crop, x0:x0 + w_crop]
    return img_rotated


def random_rotate(img, angle_vari, p_crop):
    angle = np.random.uniform(-angle_vari, angle_vari)
    crop = False if np.random.random() > p_crop else True
    return rotate_image(img, angle, crop)


def augment_images(filelist, cfg):
    for n, filepath in enumerate(filelist):
        img = cv2.imread(filepath)
        if img.shape[:2] != cfg.img_size:
            img = cv2.resize(img, cfg.img_size)
        filename = filepath.split(os.sep)[-1]
        dot_pos = filename.rfind('.')
        imgname = filename[:dot_pos]
        ext = filename[dot_pos:]

        print('Augmenting {} ...'.format(filename))
        for i in range(n):
            img_varied = img.copy()
            varied_imgname = '{}_{:0>3d}_'.format(imgname, i)

            if random.random() < cfg.p_rotate:
                img_varied_ = random_rotate(
                    img_varied,
                    cfg.rotate_angle_vari,
                    cfg.p_rotate_crop)
                if img_varied_.shape[0] >= cfg.crop_size[0] and img_varied_.shape[1] >= cfg.crop_size[1]:
                    img_varied = img_varied_
                varied_imgname += 'r'

            if random.random() < cfg.p_crop:
                img_varied = random_crop(
                    img_varied,
                    cfg.crop_size)
                varied_imgname += 'c'

            if random.random() < cfg.p_horizontal_flip:
                img_varied = cv2.flip(img_varied, 1)
                varied_imgname += 'h'

            if random.random() < cfg.p_vertical_flip:
                img_varied = cv2.flip(img_varied, 0)
                varied_imgname += 'v'

            output_filepath = os.sep.join(["C:/Users/ricsi/Desktop/aug", '{}{}'.format(varied_imgname, ext)])
            cv2.imwrite(output_filepath, img_varied)


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
    patches = patches.detach().numpy()
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
    :param type_of_network:
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
