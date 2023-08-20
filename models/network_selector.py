import torch

from abc import ABC, abstractmethod

from models.base_model import AutoEncoder
from models.extended_model import ExtendedAutoEncoder


class BaseNetwork(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def forward(self, x):
        pass


class BaseModelWrapper(BaseNetwork):
    def __init__(self, network_cfg):
        self.model = AutoEncoder(network_cfg)

    def forward(self, x):
        return self.model(x)


class ExtendedModelWrapper(BaseNetwork):
    def __init__(self, network_cfg):
        self.model = ExtendedAutoEncoder(network_cfg)

    def forward(self, x):
        return self.model(x)


class NetworkFactory:
    @staticmethod
    def create_network(network_type, network_cfg, device=None):
        if network_type == "BASE":
            model = BaseModelWrapper(network_cfg).model
        elif network_type == "EXTENDED":
            model = ExtendedModelWrapper(network_cfg).model
        else:
            raise ValueError("Wrong network type")

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model.to(device)
        return model
