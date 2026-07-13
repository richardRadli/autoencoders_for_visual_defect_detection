from pathlib import Path

from shared.core.path_bindings import dataset_paths, training_testing_paths
from utils.system_utils import (
    file_reader,
    find_latest_file_in_latest_directory,
    find_latest_valid_run,
    list_subdirectories,
    read_json_safely,
    sample_evenly,
    safe_image_path,
)


NETWORK_TYPES = ("AE", "AEE", "DAE", "DAEE")
OUTPUT_PREVIEW_TYPES = (
    "reconstruction",
    "reconstruction_vis",
    "roc_plot",
)


class DatasetService:
    """
    Provide readiness and preview facts for defect-detection datasets.

    The service contains dataset and filesystem selection logic. The API only
    validates HTTP parameters and translates service errors to HTTP responses.
    """

    @staticmethod
    def get_readiness(dataset_type: str) -> dict:
        """
        Report available training inputs and trained network types.

        Augmentation and noise facts come from their latest completed,
        non-stopped runs. A network is listed as trained only when the model
        selection used by testing finds a weight and the required training
        metadata is available beside it.

        Args:
            dataset_type: Selected dataset name.

        Returns:
            dict: Flat readiness facts for the frontend workflow.
        """
        paths = dataset_paths(dataset_type)

        aug_run = find_latest_valid_run(paths["aug"])
        aug_images = (
            len(file_reader(str(aug_run), "png", "jpg"))
            if aug_run else 0
        )

        noise_run = find_latest_valid_run(paths["noise"])
        noise_images = (
            len(file_reader(str(noise_run), "png", "jpg"))
            if noise_run else 0
        )

        trained_networks = [
            network_type
            for network_type in NETWORK_TYPES
            if DatasetService._has_weights(dataset_type, network_type)
        ]

        return {
            "dataset_type": dataset_type,
            "aug": {
                "ready": aug_images > 0,
                "images": aug_images,
            },
            "noise": {
                "ready": noise_images > 0,
                "images": noise_images,
            },
            "trained_networks": trained_networks,
        }

    @staticmethod
    def _has_weights(dataset_type: str, network_type: str) -> bool:
        """
        Check whether testing can find a weight and its required metadata.

        Args:
            dataset_type: Selected dataset name.
            network_type: Network type to inspect.

        Returns:
            bool: True when the latest selected run contains a .pt file and a
            readable params.json with valid grayscale and latent-space values.
        """
        weights_root = (
            training_testing_paths(dataset_type)["model_weights"]
            / network_type
        )

        if not weights_root.is_dir():
            return False

        try:
            weights_path = Path(
                find_latest_file_in_latest_directory(
                    str(weights_root),
                    extension=".pt",
                )
            )
            params = read_json_safely(weights_path.parent / "params.json")
        except (OSError, ValueError):
            return False

        if not isinstance(params, dict):
            return False

        grayscale = params.get("grayscale")
        latent_space_dimension = params.get("latent_space_dimension")

        return (
            isinstance(grayscale, bool)
            and isinstance(latent_space_dimension, int)
            and not isinstance(latent_space_dimension, bool)
            and latent_space_dimension >= 1
        )

    @staticmethod
    def _resolve(
        dataset_type: str,
        preview_type: str,
        subtest_folder: str | None,
        network_type: str | None,
    ) -> tuple[Path, Path | None]:
        """
        Resolve the root and image directory for a preview request.

        Test images live directly in their subtest folder. Generated outputs
        live under preview-type/network-type/timestamp directories.

        Args:
            dataset_type: Selected dataset name.
            preview_type: Requested preview image type.
            subtest_folder: Test subset required for test previews.
            network_type: Network required for generated-output previews.

        Returns:
            tuple[Path, Path | None]: The allowed root and the directory whose
            images should be listed. The source directory is None when no output
            exists.

        Raises:
            ValueError: If the preview type or one of its required selectors is
            invalid.
        """
        if preview_type == "test":
            test_directories = dataset_paths(dataset_type)["test"]

            if subtest_folder not in test_directories:
                raise ValueError(
                    f"Invalid subtest_folder '{subtest_folder}' "
                    f"for dataset '{dataset_type}'"
                )

            root = test_directories[subtest_folder]
            return root, root if root.is_dir() else None

        if preview_type in OUTPUT_PREVIEW_TYPES:
            if network_type not in NETWORK_TYPES:
                raise ValueError(
                    f"Invalid network_type '{network_type}' "
                    f"for preview_type '{preview_type}'"
                )

            root = (
                training_testing_paths(dataset_type)[preview_type]
                / network_type
            )
            runs = list_subdirectories(root)
            source_dir = runs[-1] if runs else None
            return root, source_dir

        raise ValueError(
            f"Invalid preview_type '{preview_type}'"
        )

    @staticmethod
    def get_preview_list(
        dataset_type: str,
        preview_type: str,
        subtest_folder: str | None = None,
        network_type: str | None = None,
        limit: int = 5,
    ) -> list[str]:
        """
        Return evenly sampled preview image paths.

        The returned paths are relative to the preview root and can be passed
        directly to resolve_preview_image. The API controls the allowed limit;
        this method uses the received value unchanged.

        Args:
            dataset_type: Selected dataset name.
            preview_type: Requested preview image type.
            subtest_folder: Test subset required for test previews.
            network_type: Network required for generated-output previews.
            limit: Maximum number of images requested by the API.

        Returns:
            list[str]: Root-relative image paths in POSIX format. Returns an
            empty list when the selected source directory has no images.

        Raises:
            ValueError: If the preview type or a required selector is invalid.
        """
        root, source_dir = DatasetService._resolve(
            dataset_type,
            preview_type,
            subtest_folder,
            network_type,
        )

        if source_dir is None:
            return []

        images = file_reader(
            str(source_dir),
            "png",
            "jpg",
        )
        sampled_images = sample_evenly(images, limit)

        return [
            Path(image).relative_to(root).as_posix()
            for image in sampled_images
        ]

    @staticmethod
    def resolve_preview_image(
        dataset_type: str,
        preview_type: str,
        name: str,
        subtest_folder: str | None = None,
        network_type: str | None = None,
    ) -> Path:
        """
        Resolve and validate one preview image.

        Args:
            dataset_type: Selected dataset name.
            preview_type: Requested preview image type.
            name: Root-relative path returned by get_preview_list.
            subtest_folder: Test subset required for test previews.
            network_type: Network required for generated-output previews.

        Returns:
            Path: Validated absolute image path.

        Raises:
            ValueError: If the selectors are invalid, the path escapes its
            allowed root, the extension is unsupported, or the file is missing.
        """
        root, _ = DatasetService._resolve(
            dataset_type,
            preview_type,
            subtest_folder,
            network_type,
        )
        return safe_image_path(root, name)