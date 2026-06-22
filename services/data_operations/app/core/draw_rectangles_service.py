import logging
import os
import random
import cv2

from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from colorthief import ColorThief

from services.data_operations.app.core.aug_config_service import AugmentationConfig
from shared.core.path_bindings import dataset_paths
from utils.utils import numerical_sort, resolve_num_workers

SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}

@dataclass
class DrawRectanglesResult:
    """Summary of a draw-rectangles run."""

    dataset_type: str
    source_dir: Path
    target_dir: Path
    processed_images: int


def _read_image_paths(source_dir: Path) -> list[str]:
    """
    Read supported image paths from a directory with case-insensitive extensions.

    Args:
        source_dir: Directory containing input images.

    Returns:
        list[str]: Numerically sorted image paths.
    """
    image_paths = [
        str(path)
        for path in source_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    ]

    return sorted(image_paths, key=numerical_sort)

class DrawRectanglesService:
    @staticmethod
    def _process_image(image_path: str, target_dir: str, crop_size: int, size_of_cover: int) -> None:
        """
        Draw a dominant-color square on one image and save it to the target directory.

        Args:
            image_path: Source image path.
            target_dir: Directory where the processed image is saved.
            crop_size: Size used to limit the random square position.
            size_of_cover: Width and height of the square.

        Returns:
            None
        """
        image = cv2.imread(image_path, 1)
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")

        dominant_color = ColorThief(image_path).get_color(quality=1)

        max_position = crop_size - size_of_cover
        if max_position < 0:
            raise ValueError("size_of_cover cannot be larger than crop_size")

        rand_x = random.randint(0, max_position)
        rand_y = random.randint(0, max_position)

        covered_image = cv2.rectangle(
            img=image,
            pt1=(rand_x, rand_y),
            pt2=(rand_x + size_of_cover, rand_y + size_of_cover),
            color=dominant_color,
            thickness=-1,
        )

        name = os.path.splitext(os.path.basename(image_path))[0]
        target_path = os.path.join(target_dir, f"{name}.jpg")
        cv2.imwrite(target_path, covered_image)



    @staticmethod
    def run(dataset_type: str, config: AugmentationConfig) -> DrawRectanglesResult:
        """
        Generate noisy rectangle images for a dataset.

        Args:
            dataset_type: Selected dataset name.
            config: Loaded augmentation configuration.

        Returns:
            DrawRectanglesResult: Summary of the processing.
        """
        paths = dataset_paths(dataset_type)
        source_dir = paths["good"]
        target_dir = paths["noise"]

        target_dir.mkdir(parents=True, exist_ok=True)

        image_paths = _read_image_paths(source_dir)

        num_workers = resolve_num_workers(config.num_workers)

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(
                    DrawRectanglesService._process_image,
                    image_path,
                    str(target_dir),
                    config.crop_size,
                    config.size_of_cover,
                )
                for image_path in image_paths
            ]

            for future in tqdm(futures, desc="Processing images", total=len(futures)):
                future.result()

        logging.info(f"Processed {len(image_paths)} images for dataset: {dataset_type}")

        return DrawRectanglesResult(
            dataset_type=dataset_type,
            source_dir=source_dir,
            target_dir=target_dir,
            processed_images=len(image_paths),
        )