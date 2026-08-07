import json
import logging
import os

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class TuningConfig:
    n_trials: int
    epochs_per_trial: int
    learning_rate_min: float
    learning_rate_max: float
    latent_space_dimension_min: int
    latent_space_dimension_max: int
    step_size_min: int
    step_size_max: int
    gamma_min: float
    gamma_max: float
    batch_size_min: int
    batch_size_max: int


class TuningConfigService:

    @staticmethod
    def load(base_path: str) -> TuningConfig:
        base_path = str(base_path)
        if os.path.isdir(base_path):
            target_file = os.path.join(base_path, "tuning_config.json")
        elif not base_path.endswith(".json"):
            target_file = f"{base_path}.json"
        else:
            target_file = base_path

        try:
            with open(target_file, "r", encoding="utf-8") as f:
                data: Dict[str, Any] = json.load(f)

            tuning_config = TuningConfig(
                n_trials=data["n_trials"],
                epochs_per_trial=data["epochs_per_trial"],
                learning_rate_min=data["learning_rate_min"],
                learning_rate_max=data["learning_rate_max"],
                latent_space_dimension_min=data["latent_space_dimension_min"],
                latent_space_dimension_max=data["latent_space_dimension_max"],
                step_size_min=data["step_size_min"],
                step_size_max=data["step_size_max"],
                gamma_min=data["gamma_min"],
                gamma_max=data["gamma_max"],
                batch_size_min=data["batch_size_min"],
                batch_size_max=data["batch_size_max"],
            )

            logging.info(f"Loaded tuning config from: {target_file}")
            return tuning_config

        except KeyError as e:
            logging.error(f"Missing config field: {e}")
            raise

        except Exception as e:
            logging.error(f"Failed to load tuning config: {e}")
            raise