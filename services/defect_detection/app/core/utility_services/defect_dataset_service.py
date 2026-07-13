from pathlib import Path

from shared.core.path_bindings import dataset_paths, training_testing_paths
from utils.system_utils import file_reader, find_latest_valid_run, sample_evenly, safe_image_path


NETWORK_TYPES = ("AE", "AEE", "DAE", "DAEE")


class DatasetService:
    """Readiness and preview facts for the defect_detection datasets."""

    @staticmethod
    def get_readiness(dataset_type: str) -> dict:
        """
        Report the training inputs and which networks have trained weights.

        Flat facts only; the frontend derives training/testing readiness for the
        selected network (AE/AEE need aug; DAE/DAEE need aug + noise; testing
        needs a trained model).

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

        trained = [
            network_type for network_type in NETWORK_TYPES
            if DatasetService._has_weights(dataset_type, network_type)
        ]

        return {
            "dataset_type": dataset_type,
            "aug": {"ready": aug_images > 0, "images": aug_images},
            "noise": {"ready": noise_images > 0, "images": noise_images},
            "trained_networks": trained,
        }

    @staticmethod
    def _has_weights(dataset_type: str, network_type: str) -> bool:
        """
        Check whether any trained .pt weight exists for a network.

        Args:
            dataset_type: Selected dataset name.
            network_type: One of AE / AEE / DAE / DAEE.

        Returns:
            bool: True if at least one .pt file exists for this network.
        """
        root = training_testing_paths(dataset_type)["model_weights"] / network_type
        return root.is_dir() and any(root.rglob("*.pt"))

    @staticmethod
    def get_preview_list(dataset_type: str, subtest_folder: str, limit: int = 5) -> list[str]:
        """
        List evenly sampled test images for a subtest folder.

        Args:
            dataset_type: Selected dataset name.
            subtest_folder: Test subset (texture: defective; cpu: added/contamination/missing).
            limit: Maximum number of images to return.

        Returns:
            list[str]: Root-relative image paths; empty if the folder is missing/empty.

        Raises:
            ValueError: If subtest_folder is not valid for the dataset.
        """
        test_map = dataset_paths(dataset_type)["test"]
        if subtest_folder not in test_map:
            raise ValueError(f"Invalid subtest_folder '{subtest_folder}' for dataset '{dataset_type}'")

        root = test_map[subtest_folder]
        if not root.is_dir():
            return []

        images = file_reader(str(root), "png", "jpg")
        sampled = sample_evenly(images, limit)
        return [Path(image).relative_to(root).as_posix() for image in sampled]

    @staticmethod
    def resolve_preview_image(dataset_type: str, subtest_folder: str, name: str) -> Path:
        """
        Resolve a test preview image to a safe absolute path under its folder.

        Args:
            dataset_type: Selected dataset name.
            subtest_folder: Test subset the image is from.
            name: Root-relative path returned by the preview list.

        Returns:
            Path: The validated absolute image path.

        Raises:
            ValueError: If subtest_folder is invalid, or the path escapes the root,
            is not an image, or does not exist.
        """
        test_map = dataset_paths(dataset_type)["test"]
        if subtest_folder not in test_map:
            raise ValueError(f"Invalid subtest_folder '{subtest_folder}' for dataset '{dataset_type}'")

        root = test_map[subtest_folder]
        return safe_image_path(root, name)