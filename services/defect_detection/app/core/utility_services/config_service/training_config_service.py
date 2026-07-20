import json
import logging
import os

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class TrainingConfig:
    validation_split: float
    network_type: str
    dataset_type: str
    epochs: int
    batch_size: int
    learning_rate: float
    decrease_learning_rate: bool
    step_size: int
    gamma: float
    grayscale: bool
    latent_space_dimension: int
    vis_during_training: bool
    vis_interval: int
    early_stopping: int
    seed: bool


class TrainingConfigService:

    @staticmethod
    def load(base_path: str) -> TrainingConfig:
        base_path = str(base_path)
        if os.path.isdir(base_path):
            target_file = os.path.join(base_path, "training_config.json")
        elif not base_path.endswith(".json"):
            target_file = f"{base_path}.json"
        else:
            target_file = base_path

        try:
            with open(target_file, "r", encoding="utf-8") as f:
                data: Dict[str, Any] = json.load(f)

            training_config = TrainingConfig(
                validation_split=data["validation_split"],
                network_type=data["network_type"],
                dataset_type=data["dataset_type"],
                epochs=data["epochs"],
                batch_size=data["batch_size"],
                learning_rate=data["learning_rate"],
                decrease_learning_rate=data["decrease_learning_rate"],
                step_size=data["step_size"],
                gamma=data["gamma"],
                grayscale=data["grayscale"],
                latent_space_dimension=data["latent_space_dimension"],
                vis_during_training=data["vis_during_training"],
                vis_interval=data["vis_interval"],
                early_stopping=data["early_stopping"],
                seed=data["seed"],
            )

            logging.info(f"Loaded training config from: {target_file}")
            return training_config

        except KeyError as e:
            logging.error(f"Missing config field: {e}")
            raise

        except Exception as e:
            logging.error(f"Failed to load training config: {e}")
            raise