from pathlib import Path

from shared.core.path_bindings import dataset_paths
from utils.system_utils import file_reader, find_latest_valid_run, sample_evenly, safe_image_path


PREVIEW_TYPES = ("good", "aug", "noise")


class DatasetService:
    """Readiness and preview facts for the data_operations datasets."""

    @staticmethod
    def get_readiness(dataset_type: str) -> dict:
        """
        Report whether augmentation and draw-rectangles have a finished run.

        A step is ready when its latest non-stopped run exists and holds at
        least one image. Returns flat facts; the workflow logic lives on the
        frontend.

        Args:
            dataset_type: Selected dataset name.

        Returns:
            dict: Readiness facts for the workflow bar.
        """
        paths = dataset_paths(dataset_type)

        aug_run = find_latest_valid_run(paths["aug"])
        aug_images = len(file_reader(str(aug_run), "png", "jpg")) if aug_run else 0

        noise_run = find_latest_valid_run(paths["noise"])
        noise_images = len(file_reader(str(noise_run), "png", "jpg")) if noise_run else 0

        return {
            "dataset_type": dataset_type,
            "augmentation": {"ready": aug_images > 0, "images": aug_images},
            "draw_rectangles": {"ready": noise_images > 0, "images": noise_images},
        }

    @staticmethod
    def _source_dir(dataset_type: str, preview_type: str) -> Path | None:
        """
        Resolve the folder a preview type reads images from.

        'good' is a flat folder; 'aug' and 'noise' use their latest non-stopped
        run folder.

        Args:
            dataset_type: Selected dataset name.
            preview_type: One of good / aug / noise.

        Returns:
            Path | None: The folder to list, or None if there is no valid run.
        """
        root = dataset_paths(dataset_type)[preview_type]
        if preview_type == "good":
            return root if root.is_dir() else None
        return find_latest_valid_run(root)

    @staticmethod
    def get_preview_list(dataset_type: str, preview_type: str, limit: int = 5) -> list[str]:
        """
        List up to 'limit' evenly sampled preview images for a preview type.

        The returned paths are relative to the preview type's root folder, so
        the image endpoint can resolve them safely.

        Args:
            dataset_type: Selected dataset name.
            preview_type: One of good / aug / noise.
            limit: Maximum number of images to return.

        Returns:
            list[str]: Root-relative image paths (POSIX style); empty if none.

        Raises:
            ValueError: If preview_type is not a valid data_operations type.
        """
        if preview_type not in PREVIEW_TYPES:
            raise ValueError(f"Invalid preview type '{preview_type}' for data_operations")

        source_dir = DatasetService._source_dir(dataset_type, preview_type)
        if source_dir is None:
            return []

        root = dataset_paths(dataset_type)[preview_type]
        images = file_reader(str(source_dir), "png", "jpg")
        sampled = sample_evenly(images, limit)

        return [Path(image).relative_to(root).as_posix() for image in sampled]

    @staticmethod
    def resolve_preview_image(dataset_type: str, preview_type: str, name: str) -> Path:
        """
        Resolve a preview image to a safe absolute path under its root.

        Args:
            dataset_type: Selected dataset name.
            preview_type: One of good / aug / noise.
            name: Root-relative path returned by get_preview_list.

        Returns:
            Path: The validated absolute image path.

        Raises:
            ValueError: If preview_type is invalid, or the path escapes the root,
            is not an image, or does not exist.
        """
        if preview_type not in PREVIEW_TYPES:
            raise ValueError(f"Invalid preview type '{preview_type}' for data_operations")

        root = dataset_paths(dataset_type)[preview_type]
        return safe_image_path(root, name)