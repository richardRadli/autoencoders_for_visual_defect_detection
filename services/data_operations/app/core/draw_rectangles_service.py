import logging
import os
import json
import threading
import random
import cv2

from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from colorthief import ColorThief

from services.data_operations.app.core.aug_config_service import AugmentationConfig
from shared.core.path_bindings import dataset_paths
from utils.system_utils import file_reader, resolve_num_workers, create_timestamp, find_latest_directory


@dataclass
class DrawRectanglesResult:
    """Summary of a draw-rectangles run."""

    dataset_type: str
    source_dir: Path
    target_dir: Path
    processed_images: int
    stopped: bool


class DrawRectanglesService:
    """Draw a dominant-color square on the source images of one dataset."""

    _stop_event = threading.Event()

    @staticmethod
    def request_stop() -> None:
        """Signal the running draw-rectangles job to stop as soon as possible."""
        DrawRectanglesService._stop_event.set()

    @staticmethod
    def resolve_source_dir(dataset_type: str, source: str) -> Path:
        """
        Resolve the source directory to read images from.

        Args:
            dataset_type: Selected dataset name.
            source: Source folder key ("good" or "aug").

        Returns:
            Path: The good folder, or the latest augmentation timestamp folder.

        Raises:
            ValueError: If source is "aug" but no augmentation run exists yet.
        """
        paths = dataset_paths(dataset_type)
        if source == "aug":
            return Path(find_latest_directory(str(paths["aug"])))
        return paths["good"]

    @staticmethod
    def _process_image(image_path: str, target_dir: str, size_of_cover: int) -> None:
        """
        Draw a dominant-color square on one image and save it to the target directory.

        The square is placed at a random position that keeps it fully inside the
        image, computed from the loaded image's actual width and height.

        Args:
            image_path: Source image path.
            target_dir: Directory where the processed image is saved.
            size_of_cover: Width and height (in pixels) of the square.

        Returns:
            None

        Raises:
            ValueError: If the image cannot be read, or size_of_cover is larger
                than the image's width or height.
        """
        image = cv2.imread(image_path, 1)
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")

        dominant_color = ColorThief(image_path).get_color(quality=1)

        height, width = image.shape[:2]
        max_x = width - size_of_cover
        max_y = height - size_of_cover
        if max_x < 0 or max_y < 0:
            raise ValueError(
                f"size_of_cover ({size_of_cover}) is larger than the image "
                f"({width}x{height}): {image_path}"
            )

        rand_x = random.randint(0, max_x)
        rand_y = random.randint(0, max_y)

        covered_image = cv2.rectangle(
            img=image,
            pt1=(rand_x, rand_y),
            pt2=(rand_x + size_of_cover - 1,
                 rand_y + size_of_cover - 1),
            color=dominant_color,
            thickness=-1,
        )

        name = os.path.splitext(os.path.basename(image_path))[0]
        target_path = os.path.join(target_dir, f"{name}.jpg")
        cv2.imwrite(target_path, covered_image)

    @staticmethod
    def run(dataset_type: str, source: str, source_dir: Path, config: AugmentationConfig, request_params: dict) -> DrawRectanglesResult:
        """
        Generate noisy rectangle images for a dataset.

        Reads every image from source_dir, draws a random dominant-color square
        on each in parallel, and writes them to a new timestamped noise run
        folder together with a params.json describing the run.

        Args:
            dataset_type: Selected dataset name.
            source: Source folder key used ("good" or "aug").
            source_dir: Resolved directory to read images from.
            config: Loaded augmentation configuration.
            request_params: Query parameters of the run, saved into params.json.

        Returns:
            DrawRectanglesResult: Summary of the processing (counts, paths, stopped).
        """
        DrawRectanglesService._stop_event.clear()

        paths = dataset_paths(dataset_type)
        target_dir = paths["noise"] / create_timestamp()
        os.makedirs(target_dir, exist_ok=True)

        image_paths = file_reader(str(source_dir), "png", "jpg")
        num_workers = resolve_num_workers(config.num_workers)

        processed = 0
        stopped = False
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(
                    DrawRectanglesService._process_image,
                    image_path,
                    str(target_dir),
                    config.size_of_cover,
                )
                for image_path in image_paths
            ]

            for future in tqdm(futures, desc="Processing images", total=len(futures)):
                if DrawRectanglesService._stop_event.is_set():
                    stopped = True
                    for f in futures:
                        f.cancel()
                    break
                future.result()
                processed += 1

        run_log = {
            "params": request_params,
            "result": {
                "stopped": stopped,
                "processed_images": processed,
            },
        }
        with open(target_dir / "params.json", "w", encoding="utf-8") as f:
            json.dump(run_log, f, indent=2)

        logging.info(
            f"Dataset {dataset_type}: {'STOPPED after' if stopped else 'processed'} "
            f"{processed}/{len(image_paths)} images from '{source}'"
        )

        return DrawRectanglesResult(
            dataset_type=dataset_type,
            source_dir=source_dir,
            target_dir=target_dir,
            processed_images=processed,
            stopped=stopped,
        )