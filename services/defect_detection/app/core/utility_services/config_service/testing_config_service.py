import json
import logging
import os

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class TestingConfig:
    network_type: str
    dataset_type: str
    subtest_folder: str
    vis_results: bool
    vis_reconstruction: bool
    grayscale: bool
    img_size: int
    crop_size: int
    stride: int
    threshold_init: float
    threshold_end: float
    num_of_steps: int

class TestingConfigService:
    @staticmethod
    def load(base_path: str) -> TestingConfig:
        base_path = str(base_path)
        if os.path.isdir(base_path):
            target_file = os.path.join(base_path, "testing_config.json")
        elif not base_path.endswith(".json"):
            target_file = f"{base_path}.json"
        else:
            target_file = base_path

        try:
            with open(target_file, "r", encoding="utf-8") as f:
                data: Dict[str, Any] = json.load(f)

            testing_config = TestingConfig(
                network_type=data["network_type"],
                dataset_type=data["dataset_type"],
                subtest_folder=data["subtest_folder"],
                vis_results=data["vis_results"],
                vis_reconstruction=data["vis_reconstruction"],
                grayscale=data["grayscale"],
                img_size=data["img_size"],
                crop_size=data["crop_size"],
                stride=data["stride"],
                threshold_init=data["threshold_init"],
                threshold_end=data["threshold_end"],
                num_of_steps=data["num_of_steps"]
            )

            logging.info(f"Loaded testing config from: {target_file}")
            return testing_config

        except KeyError as e:
            logging.error(f"Missing config field: {e}")
            raise

        except Exception as e:
            logging.error(f"Failed to load testing config: {e}")
            raise