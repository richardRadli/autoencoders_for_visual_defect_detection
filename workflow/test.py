import matplotlib.pyplot as plt
import numpy as np
import torch

from PIL import Image
from torchvision import transforms

from config.config import ConfigTesting
from config.network_config import network_configs
from models.network_selector import NetworkFactory
from utils.utils import setup_logger, use_gpu_if_available


class TestAutoEncoder:
    def __init__(self):
        self.logger = setup_logger()
        self.test_cfg = ConfigTesting().parse()
        network_cfg = network_configs().get(self.test_cfg.network_type)

        self.device = use_gpu_if_available()
        self.model = NetworkFactory.create_network(network_type=self.test_cfg.network_type,
                                                   network_cfg=network_cfg,
                                                   device="cuda")

        state_dict = torch.load("D:/AE/storage/weights/bottle_model_weights/BASE/2023-12-24_11-56-18/epoch_79.pt")
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def test_model(self):
        img = Image.open("D:/mvtec/bottle/test/contamination/000.png")

        transform = transforms.Compose(
            [
                transforms.Resize(size=self.test_cfg.img_size),
                transforms.ToTensor()
            ]
        )

        with torch.inference_mode():
            input_image = transform(img).unsqueeze(dim=0).to(self.device)
            reconstructed = self.model(input_image)

        reconstructed_np = (reconstructed.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

        plt.imshow(reconstructed_np)
        plt.axis('off')
        plt.show()


if __name__ == '__main__':
    autoencoder = TestAutoEncoder()
    autoencoder.test_model()
