import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class AugmentationConfig:
    dataset_type: str
    img_size: int
    crop_size: int
    rotate_count: int
    rotate_angle_vari: float
    p_rotate_crop: float
    horizontal_flip_count: int
    vertical_flip_count: int
    size_of_cover: int
    num_workers: int


class AugmentationConfigService:

    @staticmethod
    def load(base_path: str) -> AugmentationConfig:
        base_path = str(base_path)
        if os.path.isdir(base_path):
            target_file = os.path.join(base_path, "augmentation_config.json")
        elif not base_path.endswith(".json"):
            target_file = f"{base_path}.json"
        else:
            target_file = base_path

        try:
            with open(target_file, "r", encoding="utf-8") as f:
                data: Dict[str, Any] = json.load(f)

            augmentation_config = AugmentationConfig(
                dataset_type=data["dataset_type"],
                img_size=data["img_size"],
                crop_size=data["crop_size"],
                rotate_count=data["rotate_count"],
                rotate_angle_vari=data["rotate_angle_vari"],
                p_rotate_crop=data["p_rotate_crop"],
                horizontal_flip_count=data["horizontal_flip_count"],
                vertical_flip_count=data["vertical_flip_count"],
                size_of_cover=data["size_of_cover"],
                num_workers=data["num_workers"]

            )

            logging.info(f"Loaded config from: {target_file}")
            return augmentation_config

        except KeyError as e:
            logging.error(f"Missing config field: {e}")
            raise

        except Exception as e:
            logging.error(f"Failed to load augmentation config: {e}")
            raise