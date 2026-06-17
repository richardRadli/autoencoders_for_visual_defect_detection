import logging
import os

from pathlib import Path


# Project root (this file lives in shared/core, so two levels up is the root).
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Data and storage roots come from environment variables (set in Docker),
# falling back to local folders when not provided.
DATASET_ROOT = Path(os.getenv("DATASET_ROOT", PROJECT_ROOT / "dataset"))
STORAGE_ROOT = Path(os.getenv("STORAGE_ROOT", PROJECT_ROOT / "storage"))


class PathGroup:
    """
    Group related paths under a common root and create them on demand.

    Args:
        root: The root directory the paths are relative to.
        mapping: A dictionary of name -> relative path pairs.
    """

    def __init__(self, root: Path, mapping: dict):
        self.root = root
        self.mapping = mapping

    def __getitem__(self, key: str) -> Path:
        """
        Allow bracket access, e.g. DATASET_PATHS["texture_1_good"].

        Args:
            key: The name of the path in the mapping.

        Returns:
            Path: The full path belonging to the key.
        """
        return self.get(key)

    def get(self, key: str) -> Path:
        """
        Return the full path belonging to a given key.

        Args:
            key: The name of the path in the mapping.

        Returns:
            Path: The full path (root joined with the relative path).
        """
        path = self.mapping.get(key)
        if path is None:
            raise KeyError(f"Invalid key: {key}")
        return self.root / path

    def create_dirs(self) -> None:
        """
        Create every directory in the mapping if it does not exist yet.

        If a mapped path points to a file (it has an extension), only its
        parent directory is created, so file paths never become folders.

        Returns:
            None
        """
        for path in self.mapping.values():
            full_path = self.root / path
            # If it has an extension it's a file -> create only its parent folder.
            dir_path = full_path.parent if full_path.suffix else full_path
            dir_path.mkdir(parents=True, exist_ok=True)
            logging.info(f"Created directory {dir_path}")


# Config JSON files (reused from the existing project, relative to the root).
CONFIG_PATHS = PathGroup(
    root=STORAGE_ROOT,
    mapping={
        "augmentation_config": "config/json_files/augmentation_config",
        #"augmentation_config_schema": "config/json_files/augmentation_config_schema.json",
        #TODO SChema torlese, mert dataclass miatt nem kell
    },
)


# Dataset folders
#TODO kell még cpunal van még 3 almappa, cpu a:added ,c:contemination ,m:missing ,
DATASET_PATHS = PathGroup(
    root=DATASET_ROOT,
    mapping={
        "texture_1_good": "texture_1/train/good",
        "texture_1_aug": "texture_1/aug",
        "texture_1_noise": "texture_1/noise",
        "texture_1_test": "texture_1/test/defective/test_images",
        "texture_1_ground_truth": "texture_1/test/defective/ground_truth",

        "texture_2_good": "texture_2/train/good",
        "texture_2_aug": "texture_2/aug",
        "texture_2_noise": "texture_2/noise",
        "texture_2_test": "texture_2/test/defective/test_images",
        "texture_2_ground_truth": "texture_2/test/defective/ground_truth",

        "cpu_good": "cpu/train/good",
        "cpu_aug": "cpu/aug",
        "cpu_noise": "cpu/noise",

        "cpu_added_test": "cpu/test/cpua/test_images",
        "cpu_added_ground_truth": "cpu/test/cpua/ground_truth",

        "cpu_contamination_test": "cpu/test/cpuc/test_images",
        "cpu_contamination_ground_truth": "cpu/test/cpuc/ground_truth",

        "cpu_missing_test": "cpu/test/cpum/test_images",
        "cpu_missing_ground_truth": "cpu/test/cpum/ground_truth",
    },
)


def init_storage() -> None:
    """
    Create the dataset and storage root directories if they do not exist.

    Returns:
        None
    """
    DATASET_ROOT.mkdir(parents=True, exist_ok=True)
    STORAGE_ROOT.mkdir(parents=True, exist_ok=True)


def init_all_paths() -> None:
    """
    Create all shared directories. Called once at service startup.

    Returns:
        None
    """
    init_storage()
    CONFIG_PATHS.create_dirs()
    DATASET_PATHS.create_dirs()
    logging.info("All shared paths initialized")