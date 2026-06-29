from enum import Enum

class DatasetType(str, Enum):
    """Datasets that the utility_services can process."""

    texture_1 = "texture_1"
    texture_2 = "texture_2"
    cpu = "cpu"