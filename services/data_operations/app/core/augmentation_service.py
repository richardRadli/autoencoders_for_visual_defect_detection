import logging
import os
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
    """Generate base crops and count-based augmented images for one dataset."""

    @staticmethod
    def _distribute(count: int, num_imgs: int) -> List[int]:
        """
        Spread a total count as evenly as possible across the source images.

        Args:
            count: Total number of variants requested for one operation.
            num_imgs: Number of source images.

        Returns:
            List[int]: Per-image variant counts; the first images take the remainder.
        """
        base = count // num_imgs
        rem = count - base * num_imgs
        return [base + 1 if i < rem else base for i in range(num_imgs)]

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
            crop: Whether to crop the black corners after rotation.

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
    def _rotate_variant(image: np.ndarray, config: AugmentationConfig, crop_size: Tuple[int, int]) -> np.ndarray:
        """
        Produce one rotated variant cropped to crop_size with no black corners.

        Args:
            image: The resized source image.
            config: Loaded augmentation configuration.
            crop_size: Target height and width for the final crop.

        Returns:
            np.ndarray: A crop_size image, falling back to a plain crop if the
            trimmed rotation is too small.
        """
        rotated = AugmentationService._random_rotate(image, config.rotate_angle_vari, config.p_rotate_crop)
        if rotated.shape[0] >= crop_size[0] and rotated.shape[1] >= crop_size[1]:
            return AugmentationService._random_crop(rotated, crop_size)
        return AugmentationService._random_crop(image, crop_size)

    @staticmethod
    def _save(target_dir: str, name: str, image: np.ndarray) -> None:
        """
        Write one image as JPG to the target directory.

        Args:
            target_dir: Directory where the image is saved.
            name: File name without extension.
            image: The image to write.

        Returns:
            None
        """
        output_path = os.path.join(target_dir, f"{name}.jpg")
        if not cv2.imwrite(output_path, image):
            raise ValueError(f"Cannot write image: {output_path}")

    @staticmethod
    def _process_image(
        filepath: str,
        rotate_n: int,
        hflip_n: int,
        vflip_n: int,
        target_dir: str,
        config: AugmentationConfig,
    ) -> None:
        """
        Read one source image once and write its base crop and augmented variants.

        Args:
            filepath: Source image path.
            rotate_n: Number of rotated variants for this image.
            hflip_n: Number of horizontally flipped variants for this image.
            vflip_n: Number of vertically flipped variants for this image.
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

        AugmentationService._save(target_dir, f"{name}_crop", AugmentationService._random_crop(image, crop_size))

        for i in range(rotate_n):
            variant = AugmentationService._rotate_variant(image, config, crop_size)
            AugmentationService._save(target_dir, f"{name}_rot_{i:04d}", variant)

        for i in range(hflip_n):
            variant = cv2.flip(AugmentationService._random_crop(image, crop_size), 1)
            AugmentationService._save(target_dir, f"{name}_hflip_{i:04d}", variant)

        for i in range(vflip_n):
            variant = cv2.flip(AugmentationService._random_crop(image, crop_size), 0)
            AugmentationService._save(target_dir, f"{name}_vflip_{i:04d}", variant)

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
        Generate the base crops and the count-based augmented set for a dataset.

        Args:
            dataset_type: Selected dataset name.
            config: Loaded augmentation configuration.
            overrides: Optional per-operation count overrides from the request.

        Returns:
            AugmentationResult: Summary of the processing.
        """
        config = AugmentationService._apply_overrides(config, overrides)

        paths = dataset_paths(dataset_type)
        source_dir = paths["good"]
        target_dir = paths["aug"]
        target_dir.mkdir(parents=True, exist_ok=True)

        image_paths = file_reader(str(source_dir), "png", "jpg")
        num_imgs = len(image_paths)
        if num_imgs == 0:
            raise ValueError(f"No source images found in: {source_dir}")

        rotate_per = AugmentationService._distribute(config.rotate_count, num_imgs)
        hflip_per = AugmentationService._distribute(config.horizontal_flip_count, num_imgs)
        vflip_per = AugmentationService._distribute(config.vertical_flip_count, num_imgs)

        num_workers = resolve_num_workers(config.num_workers)

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(
                    AugmentationService._process_image,
                    image_path,
                    rotate_per[i],
                    hflip_per[i],
                    vflip_per[i],
                    str(target_dir),
                    config,
                )
                for i, image_path in enumerate(image_paths)
            ]

            for future in tqdm(futures, total=len(futures), desc="Augmenting images"):
                future.result()

        total_aug = config.rotate_count + config.horizontal_flip_count + config.vertical_flip_count
        logging.info(
            f"Dataset {dataset_type}: {num_imgs} base crops, "
            f"{config.rotate_count} rotate, {config.horizontal_flip_count} hflip, "
            f"{config.vertical_flip_count} vflip ({total_aug} augmented total)"
        )

        return AugmentationResult(
            dataset_type=dataset_type,
            source_dir=source_dir,
            target_dir=target_dir,
            source_images=num_imgs,
            augmented_images=total_aug,
        )