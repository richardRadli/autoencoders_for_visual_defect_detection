import cv2
import math
import numpy as np
import os
import random

from typing import List, Tuple
from tqdm import tqdm

from config.data_paths import JSON_FILES_PATHS
from config.dataset_config import dataset_images_path_selector
from utils.utils import load_config_json


def generate_image_list(train_data_dir: str, augment_num: int) -> List[Tuple[str, int]]:
    """
    Generate a list of image filenames with corresponding augmentation counts.

    :param train_data_dir: The directory containing the training images.
    :param augment_num: The total number of augmentations to be performed.
    :return: A list of tuples, each containing a filename and its corresponding augmentation count.
    """

    filenames = os.listdir(train_data_dir)
    num_imgs = len(filenames)
    num_ave_aug = int(math.floor(augment_num/num_imgs))
    rem = augment_num - num_ave_aug*num_imgs
    lucky_seq = [True]*rem + [False]*(num_imgs-rem)
    random.shuffle(lucky_seq)

    img_list = [
        (os.sep.join([train_data_dir, filename]), num_ave_aug+1 if lucky else num_ave_aug)
        for filename, lucky in zip(filenames, lucky_seq)
    ]

    return img_list


def random_crop(image: np.ndarray, new_size: Tuple[int, int]) -> np.ndarray:
    """
    Perform a random crop on the input image.

    :param image: The input image as a NumPy array.
    :param new_size: A tuple representing the new size (height, width) of the cropped image.
    :return: The cropped image.
    """

    h, w = image.shape[:2]
    y = np.random.randint(0, h - new_size[0])
    x = np.random.randint(0, w - new_size[1])
    image = image[y:y + new_size[0], x:x + new_size[1]]
    return image


def rotate_image(img: np.ndarray, angle: float, crop: bool) -> np.ndarray:
    """
    Rotate the input image by the specified angle.

    :param img: The input image as a NumPy array.
    :param angle: The angle (in degrees) by which to rotate the image.
    :param crop: A boolean indicating whether to crop the image after rotation.
    :return: The rotated image.
    """

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


def random_rotate(img: np.ndarray, angle_vari: float, p_crop: float) -> np.ndarray:
    """
    Rotate the input image by a random angle within the specified range.

    :param img: The input image as a NumPy array.
    :param angle_vari: The range of variation for the random rotation angle.
    :param p_crop: The probability of cropping the image after rotation.
    :return: The randomly rotated image.
    """

    angle = np.random.uniform(-angle_vari, angle_vari)
    crop = False if np.random.random() > p_crop else True
    return rotate_image(img, angle, crop)


def augment_images(filelist: List[Tuple[str, int]], aug_out_dir: str, cfg) -> None:
    """
    Augment a list of images and save the augmented images to the specified directory.

    :param filelist: A list of tuples, each containing a filepath and its corresponding augmentation count.
    :param aug_out_dir: The directory to save the augmented images.
    :param cfg: An object containing configuration parameters for image augmentation.
    :return: None
    """

    for filepath, n in tqdm(filelist, total=len(filelist), desc='Augmenting images'):
        img = cv2.imread(filepath)
        if img.shape[:2] != cfg.get("img_size"):
            img = cv2.resize(img, cfg.get("img_size"))
        filename = filepath.split(os.sep)[-1]
        dot_pos = filename.rfind('.')
        imgname = filename[:dot_pos]
        ext = filename[dot_pos:]

        for i in range(n):
            img_varied = img.copy()
            varied_imgname = '{}_{:0>3d}_'.format(imgname, i)

            if random.random() < cfg.get("p_rotate"):
                img_varied_ = random_rotate(
                    img_varied,
                    cfg.get("rotate_angle_vari"),
                    cfg.get("p_rotate_crop"))
                if img_varied_.shape[0] >= cfg.get("crop_size")[0] and img_varied_.shape[1] >= cfg.get("crop_size")[1]:
                    img_varied = img_varied_
                varied_imgname += 'r'

            if random.random() < cfg.get("p_crop"):
                img_varied = random_crop(
                    img_varied,
                    cfg.get("crop_size"))
                varied_imgname += 'c'

            if random.random() < cfg.get("p_horizontal_flip"):
                img_varied = cv2.flip(img_varied, 1)
                varied_imgname += 'h'

            if random.random() < cfg.get("p_vertical_flip"):
                img_varied = cv2.flip(img_varied, 0)
                varied_imgname += 'v'

            output_filepath = os.sep.join([aug_out_dir, '{}{}'.format(varied_imgname, ext)])
            cv2.imwrite(output_filepath, img_varied)


def main() -> None:
    """
    Main function for data augmentation.

    Return:
         None
    """

    aug_cfg = (
        load_config_json(json_schema_filename=JSON_FILES_PATHS.get_data_path("config_schema_augmentation"),
                         json_filename=JSON_FILES_PATHS.get_data_path("config_augmentation"))
    )

    if aug_cfg.get("do_augmentation"):
        train_data_dir = dataset_images_path_selector().get(aug_cfg.get("dataset_type"), {}).get("train")
        aug_out_dir = dataset_images_path_selector().get(aug_cfg.get("dataset_type"), {}).get("aug")

        if not train_data_dir or not aug_out_dir:
            raise ValueError("Error: Missing or invalid paths for training data or augmentation output.")

        img_list = generate_image_list(train_data_dir=train_data_dir, augment_num=aug_cfg.get("augment_num"))
        augment_images(filelist=img_list, aug_out_dir=aug_out_dir, cfg=aug_cfg)


if __name__ == "__main__":
    main()
