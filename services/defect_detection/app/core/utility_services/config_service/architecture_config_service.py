import json
import logging
import os

from typing import Dict, Any

class ArchitectureConfigService:
    @staticmethod
    def build(base_path: str, network_type: str, grayscale: bool, latent_space_dimension: int) -> dict:
        """
        Build the model architecture cfg from network_config.json.

        Args:
            base_path: Folder or path of the network_config.json.
            network_type: One of AE / AEE / DAE / DAEE.
            grayscale: Whether the input is grayscale (1 channel) or RGB (3).
            latent_space_dimension: Size of the latent space.

        Returns:
            dict: The network_cfg expected by NetworkFactory.
        """
        base_path = str(base_path)
        if os.path.isdir(base_path):
            target_file = os.path.join(base_path, "network_config.json")
        elif not base_path.endswith(".json"):
            target_file = f"{base_path}.json"
        else:
            target_file = base_path

        with open(target_file, "r", encoding="utf-8") as f:
            data: Dict[str, Any] = json.load(f)

        architecture = data["architecture_type"][network_type]
        input_channel = data["input_channel"]["grayscale" if grayscale else "rgb"]

        architecture_cfg = {
            "kernel_size": data["kernel_size"],
            "stride": data["stride"],
            "padding": data["padding"],
            "flc": data["flc"][architecture],
            "alpha_slope": data["alpha_slope"],
            "latent_space_dimension": latent_space_dimension,
            "input_channel": input_channel,
        }

        logging.info(f"Built architecture cfg for {network_type} ({architecture}) from {target_file}")
        return architecture_cfg