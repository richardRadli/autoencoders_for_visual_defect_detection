import json
import logging

from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class AugmentationConfig:
        dataset_type: str
        img_size: int
        crop_size: int
        augment_num: int
        p_rotate: float
        rotate_angle_vari: float
        p_rotate_crop: float
        p_crop: int
        p_horizontal_flip: float
        p_vertical_flip: float
        size_of_cover: int
        num_workers: int

class DeviceConfigService:

    @staticmethod
    def load(filepath: str) -> AugmentationConfig:
        try:
            with open(filepath, "r") as f:
                data: Dict[str, Any] = json.load(f)

            augmentation_config = AugmentationConfig(
                dataset_type=data["dataset_type"],
                img_size=data["img_size"],
                crop_size=data["crop_size"],
                augment_num=data["augment_num"],
                p_rotate=data["p_rotate"],
                rotate_angle_vari=data["rotate_angle_vari"],
                p_rotate_crop=data["p_rotate_crop"],
                p_crop=data["p_crop"],
                p_horizontal_flip=data["p_horizontal_flip"],
                p_vertical_flip=data["p_vertical_flip"],
                size_of_cover=data["size_of_cover"],
                num_workers=data["num_workers"],
            )


            logging.info(f"Loaded device config: {filepath}")
            return augmentation_config

        except KeyError as e:
            logging.error( f"Missing config field: {e}")
            raise

        except Exception as e:
            logging.error( f"Failed to load device config: {e}")
            raise
