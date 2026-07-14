from pathlib import Path

from shared.core.path_bindings import dataset_paths
from utils.system_utils import list_subdirectories, read_json_safely


class RunResolverService:
    """Resolve and validate augmentation runs by their saved (img_size, crop_size)."""

    @staticmethod
    def _run_matches(run_dir: Path, img_size: int, crop_size: int) -> bool:
        """
        Check whether a run folder is a completed run for the given sizes.

        A run matches when its params.json exists, is a JSON object, is not
        stopped, and records exactly img_size and crop_size under "params".

        Args:
            run_dir: The run folder to inspect.
            img_size: Required image size.
            crop_size: Required crop size.

        Returns:
            bool: True if the run is completed and matches both sizes.
        """
        params = read_json_safely(run_dir / "params.json")
        if not isinstance(params, dict):
            return False

        result = params.get("result")
        if not isinstance(result, dict) or result.get("stopped") is not False:
            return False

        run_params = params.get("params")
        if not isinstance(run_params, dict):
            return False

        return run_params.get("img_size") == img_size and run_params.get("crop_size") == crop_size

    @staticmethod
    def resolve_augmentation_run(dataset_type: str, img_size: int, crop_size: int) -> str | None:
        """
        Find the latest completed augmentation run matching img_size and crop_size.

        Args:
            dataset_type: Selected dataset name.
            img_size: Required image size.
            crop_size: Required crop size.

        Returns:
            str | None: The run's relative identifier (its timestamp folder name),
            or None if no completed run matches both sizes.
        """
        aug_root = dataset_paths(dataset_type)["aug"]
        for run_dir in reversed(list_subdirectories(aug_root)):
            if RunResolverService._run_matches(run_dir, img_size, crop_size):
                return run_dir.name
        return None

    @staticmethod
    def validate_augmentation_run(dataset_type: str, run_name: str, img_size: int, crop_size: int) -> Path:
        """
        Re-validate a previously resolved augmentation run and return its path.

        Used by the training task to confirm the exact run handed to it is still
        a completed run for the requested sizes, instead of re-searching. Only a
        single, direct child folder name of the aug root is accepted — absolute
        paths, nested paths and '..' are rejected.

        Args:
            dataset_type: Selected dataset name.
            run_name: The run's relative identifier (a single timestamp folder name).
            img_size: Required image size.
            crop_size: Required crop size.

        Returns:
            Path: The validated, resolved run folder path.

        Raises:
            ValueError: If run_name is not a single direct child of the aug root,
                or the run is missing, incomplete, or does not match both sizes.
        """
        aug_root = Path(dataset_paths(dataset_type)["aug"]).resolve()
        run_path = Path(run_name)

        if run_path.is_absolute() or len(run_path.parts) != 1:
            raise ValueError(
                f"Invalid augmentation run name '{run_name}' for dataset '{dataset_type}'"
            )

        run_dir = (aug_root / run_path).resolve()
        if run_dir.parent != aug_root:
            raise ValueError(
                f"Invalid augmentation run name '{run_name}' for dataset '{dataset_type}'"
            )

        if not RunResolverService._run_matches(run_dir, img_size, crop_size):
            raise ValueError(
                f"Augmentation run '{run_name}' for dataset '{dataset_type}' is missing, "
                f"incomplete, or does not match img_size={img_size} / crop_size={crop_size}"
            )
        return run_dir