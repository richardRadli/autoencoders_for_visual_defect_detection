from abc import ABC, abstractmethod

from models.base_model import BaseAutoEncoder
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
        self.model = BaseAutoEncoder(network_cfg)

    def forward(self, x):
        return self.model(x)


class ExtendedModelWrapper(BaseNetwork):
    def __init__(self, network_cfg):
        self.model = ExtendedAutoEncoder(network_cfg)

    def forward(self, x):
        return self.model(x)


class NetworkFactory:
    _network_map = {
        "AE": BaseModelWrapper,
        "DAE": BaseModelWrapper,
        "AEE": ExtendedModelWrapper,
        "DAEE": ExtendedModelWrapper
    }

    @staticmethod
    def create_network(network_type, network_cfg):
        if network_type not in NetworkFactory._network_map:
            raise ValueError(f"Invalid network type: {network_type}")

        model_wrapper_class = NetworkFactory._network_map[network_type]
        model = model_wrapper_class(network_cfg).model

        return model
