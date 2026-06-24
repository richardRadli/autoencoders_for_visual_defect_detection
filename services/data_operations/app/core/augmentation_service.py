import logging
import math
import os
import random
import cv2
import numpy as np

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple

from shared.core.path_bindings import dataset_paths
from services.data_operations.app.core.aug_config_service import AugmentationConfig
from utils.utils import file_reader, resolve_num_workers


@dataclass
class AugmentationResult:
    """Summary of an augmentation run."""

    dataset_type: str
    source_dir: Path
    target_dir: Path
    source_images: int
    augmented_images: int


class AugmentationService:
    """Generate augmented training images for one dataset."""

    @staticmethod
    def _generate_image_list(train_data_dir: str, augment_num: int) -> List[Tuple[str, int]]:
        """
        Generate image paths with their per-image augmentation counts.

        Args:
            train_data_dir: Directory containing the training images.
            augment_num: Total number of augmentations to distribute.

        Returns:
            List[Tuple[str, int]]: Image path and augmentation count pairs.
        """
        image_paths = file_reader(train_data_dir, "png", "jpg")
        num_imgs = len(image_paths)

        if num_imgs == 0:
            raise ValueError(f"No source images found in: {train_data_dir}")

        num_ave_aug = int(math.floor(augment_num / num_imgs))
        rem = augment_num - num_ave_aug * num_imgs
        lucky_seq = [True] * rem + [False] * (num_imgs - rem)
        random.shuffle(lucky_seq)

        return [
            (image_path, num_ave_aug + 1 if lucky else num_ave_aug)
            for image_path, lucky in zip(image_paths, lucky_seq)
        ]

    @staticmethod
    def _random_crop(image: np.ndarray, new_size: Tuple[int, int]) -> np.ndarray:
        """
        Perform a random crop on the input image.

        Args:
            image: The input image as a NumPy array.
            new_size: Target height and width of the crop.

        Returns:
            np.ndarray: The cropped image.
        """
        h, w = image.shape[:2]
        y = np.random.randint(0, h - new_size[0])
        x = np.random.randint(0, w - new_size[1])
        return image[y:y + new_size[0], x:x + new_size[1]]

    @staticmethod
    def _rotate_image(img: np.ndarray, angle: float, crop: bool) -> np.ndarray:
        """
        Rotate the input image by the specified angle.

        Args:
            img: The input image as a NumPy array.
            angle: Rotation angle in degrees.
            crop: Whether to crop the image after rotation.

        Returns:
            np.ndarray: The rotated image.
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

    @staticmethod
    def _random_rotate(img: np.ndarray, angle_vari: float, p_crop: float) -> np.ndarray:
        """
        Rotate the input image by a random angle within the given range.

        Args:
            img: The input image as a NumPy array.
            angle_vari: Range of variation for the random rotation angle.
            p_crop: Positive value enables crop-after-rotation.

        Returns:
            np.ndarray: The randomly rotated image.
        """
        angle = np.random.uniform(-angle_vari, angle_vari)
        crop = p_crop > 0
        return AugmentationService._rotate_image(img, angle, crop)

    @staticmethod
    def _augment_single(
        image: np.ndarray,
        config: AugmentationConfig,
        crop_size: Tuple[int, int],
    ) -> Tuple[np.ndarray, str]:
        """
        Apply enabled transforms to a single image.

        Args:
            image: The image to transform.
            config: Loaded augmentation configuration.
            crop_size: Target height and width for cropping.

        Returns:
            Tuple[np.ndarray, str]: The transformed image and its name suffix.
        """
        suffix = ""

        if config.p_rotate > 0:
            rotated = AugmentationService._random_rotate(
                image,
                config.rotate_angle_vari,
                config.p_rotate_crop,
            )
            if rotated.shape[0] >= crop_size[0] and rotated.shape[1] >= crop_size[1]:
                image = rotated
            suffix += "r"

        if config.p_crop > 0:
            image = AugmentationService._random_crop(image, crop_size)
            suffix += "c"

        if config.p_horizontal_flip > 0:
            image = cv2.flip(image, 1)
            suffix += "h"

        if config.p_vertical_flip > 0:
            image = cv2.flip(image, 0)
            suffix += "v"

        return image, suffix

    @staticmethod
    def _process_image(filepath: str, count: int, target_dir: str, config: AugmentationConfig) -> None:
        """
        Read one source image once and write its augmented variants.

        Args:
            filepath: Source image path.
            count: Number of variants to generate.
            target_dir: Directory where the variants are saved.
            config: Loaded augmentation configuration.

        Returns:
            None
        """
        img_size = (config.img_size, config.img_size)
        crop_size = (config.crop_size, config.crop_size)

        image = cv2.imread(filepath)
        if image is None:
            raise ValueError(f"Cannot read image: {filepath}")

        if image.shape[:2] != img_size:
            image = cv2.resize(image, img_size)

        name = Path(filepath).stem
        ext = Path(filepath).suffix

        for i in range(count):
            varied, suffix = AugmentationService._augment_single(image.copy(), config, crop_size)
            output_path = os.path.join(target_dir, f"{name}_{i:03d}_{suffix}{ext}")

            if not cv2.imwrite(output_path, varied):
                raise ValueError(f"Cannot write image: {output_path}")

    @staticmethod
    def _apply_overrides(config: AugmentationConfig, overrides: dict | None) -> AugmentationConfig:
        """
        Override config fields with the provided non-None values.

        Args:
            config: Loaded augmentation configuration.
            overrides: Field name to value pairs; None values are ignored.

        Returns:
            AugmentationConfig: The config with the overrides applied.
        """
        if not overrides:
            return config

        active = {key: value for key, value in overrides.items() if value is not None}
        if not active:
            return config

        return replace(config, **active)

    @staticmethod
    def run(dataset_type: str, config: AugmentationConfig, overrides: dict | None = None) -> AugmentationResult:
        """
        Generate the augmented training set for a dataset.

        Args:
            dataset_type: Selected dataset name.
            config: Loaded augmentation configuration.
            overrides: Optional per-operation p_* overrides from the request.

        Returns:
            AugmentationResult: Summary of the processing.
        """
        config = AugmentationService._apply_overrides(config, overrides)

        paths = dataset_paths(dataset_type)
        source_dir = paths["good"]
        target_dir = paths["aug"]

        target_dir.mkdir(parents=True, exist_ok=True)

        img_list = AugmentationService._generate_image_list(str(source_dir), config.augment_num)
        num_workers = resolve_num_workers(config.num_workers)

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(
                    AugmentationService._process_image,
                    filepath,
                    count,
                    str(target_dir),
                    config,
                )
                for filepath, count in img_list
            ]

            for future in tqdm(futures, total=len(futures), desc="Augmenting images"):
                future.result()

        logging.info(f"Augmented {config.augment_num} images for dataset: {dataset_type}")

        return AugmentationResult(
            dataset_type=dataset_type,
            source_dir=source_dir,
            target_dir=target_dir,
            source_images=len(img_list),
            augmented_images=config.augment_num,
        )
