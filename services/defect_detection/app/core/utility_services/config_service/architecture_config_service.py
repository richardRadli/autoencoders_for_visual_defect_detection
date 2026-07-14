import json
import logging
import os

from typing import Any, Dict


class ArchitectureConfigService:
    @staticmethod
    def build(
        base_path: str,
        network_type: str,
        grayscale: bool,
        latent_space_dimension: int,
        crop_size: int,
    ) -> dict:
        """
        Build the model architecture cfg from network_config.json.

        The kernel/stride/padding are selected from network_by_crop_size by the
        crop size, so the final (bottleneck) kernel matches the encoder's
        feature-map size and every supported crop collapses to an L x 1 x 1
        latent. The model classes keep using kernel_size[2] unchanged.

        Args:
            base_path: Folder or path of the network_config.json.
            network_type: One of AE / AEE / DAE / DAEE.
            grayscale: Whether the input is grayscale (1 channel) or RGB (3).
            latent_space_dimension: Size of the latent space.
            crop_size: Patch size the model is built for; must be one of the
                crop sizes defined in network_by_crop_size.

        Returns:
            dict: The network_cfg expected by NetworkFactory, with the
            crop-specific kernel_size / stride / padding.

        Raises:
            ValueError: If crop_size is not a plain int, the network_type is
                unknown, or the crop_size has no entry in network_by_crop_size.
        """
        # A bool is an int subclass in Python; reject it explicitly.
        if isinstance(crop_size, bool) or not isinstance(crop_size, int):
            raise ValueError(
                f"crop_size must be an int, got {type(crop_size).__name__}: {crop_size!r}"
            )

        base_path = str(base_path)
        if os.path.isdir(base_path):
            target_file = os.path.join(base_path, "network_config.json")
        elif not base_path.endswith(".json"):
            target_file = f"{base_path}.json"
        else:
            target_file = base_path

        with open(target_file, "r", encoding="utf-8") as f:
            data: Dict[str, Any] = json.load(f)

        architecture_by_type = data["architecture_type"]
        if network_type not in architecture_by_type:
            raise ValueError(
                f"Unknown network_type '{network_type}'. Allowed: {sorted(architecture_by_type)}"
            )
        architecture = architecture_by_type[network_type]

        crop_configs = data["network_by_crop_size"]
        crop_key = str(crop_size)
        if crop_key not in crop_configs:
            raise ValueError(
                f"Unsupported crop_size {crop_size}. "
                f"Allowed: {sorted(int(key) for key in crop_configs)}"
            )
        crop_cfg = crop_configs[crop_key]

        input_channel = data["input_channel"]["grayscale" if grayscale else "rgb"]

        architecture_cfg = {
            "kernel_size": crop_cfg["kernel_size"],
            "stride": crop_cfg["stride"],
            "padding": crop_cfg["padding"],
            "flc": data["flc"][architecture],
            "alpha_slope": data["alpha_slope"],
            "latent_space_dimension": latent_space_dimension,
            "input_channel": input_channel,
        }

        logging.info(
            f"Built architecture cfg for {network_type} ({architecture}) from {target_file} "
            f"(crop_size={crop_size}, kernel_size={crop_cfg['kernel_size']})"
        )
        return architecture_cfg