import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from skimage.metrics import structural_similarity as ssim

from config.config import ConfigTesting
from config.network_config import network_configs
from models.network_selector import NetworkFactory
from utils.utils import setup_logger, use_gpu_if_available, get_patch, patch2img


class TestAutoEncoder:
    def __init__(self):
        self.logger = setup_logger()
        self.test_cfg = ConfigTesting().parse()
        network_cfg = network_configs().get(self.test_cfg.network_type)

        self.device = "cpu" #use_gpu_if_available()
        self.model = NetworkFactory.create_network(network_type=self.test_cfg.network_type,
                                                   network_cfg=network_cfg,
                                                   device=self.device)

        state_dict = torch.load("D:/AE/storage/weights/bottle_model_weights/BASE/2023-12-25_17-22-46/epoch_81.pt")
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def get_residual_map(self):
        test_img = cv2.imread("D:/mvtec/bottle/test/contamination/000.png")
        mask_size = self.test_cfg.patch_size \
            if self.test_cfg.img_size[0] - self.test_cfg.crop_size[0] < self.test_cfg.stride \
            else self.test_cfg.img_size[0]

        if test_img.shape[:2] != self.test_cfg.img_size:
            test_img = cv2.resize(test_img, self.test_cfg.img_size)
        if self.test_cfg.img_size[0] != mask_size:
            tmp = (self.test_cfg.img_size[0] - mask_size) // 2
            test_img = test_img[tmp:tmp + mask_size, tmp:tmp + mask_size]

        test_img_ = test_img / 255.

        if test_img.shape[:2] == self.test_cfg.crop_size:
            test_img_ = np.expand_dims(test_img_, 0)
            decoded_img = self.model(test_img_)
        else:
            patches = get_patch(test_img_, self.test_cfg.crop_size[0], self.test_cfg.stride)
            patches = np.transpose(patches, (0, 3, 1, 2))
            patches = torch.from_numpy(patches).float()
            patches = self.model(patches)
            decoded_img = (
                patch2img(patches, self.test_cfg.img_size[0], self.test_cfg.crop_size[0], self.test_cfg.stride)
            )

        rec_img = np.reshape((decoded_img * 255.).astype('uint8'), test_img.shape)
        _, ssim_residual_map = ssim(test_img, rec_img, full=True, multichannel=True, channel_axis=2)
        ssim_residual_map = 1 - np.mean(ssim_residual_map, axis=2)
        l1_residual_map = np.mean(np.abs(test_img / 255. - rec_img / 255.), axis=2)

        return test_img, rec_img, ssim_residual_map, l1_residual_map

    def plot_results(self):
        # Plot RGB images (test_img and rec_img)
        plt.subplot(2, 2, 1)
        plt.imshow(test_img)
        plt.title('test_img (RGB)')

        plt.subplot(2, 2, 2)
        plt.imshow(rec_img)
        plt.title('rec_img (RGB)')

        # Plot grayscale images (ssim_residual_map and l1_residual_map)
        plt.subplot(2, 2, 3)
        plt.imshow(ssim_residual_map, cmap='gray')
        plt.title('SSIM Residual Map (Grayscale)')

        plt.subplot(2, 2, 4)
        plt.imshow(l1_residual_map, cmap='gray')
        plt.title('L1 Residual Map (Grayscale)')

        # Adjust layout for better visualization
        plt.tight_layout()

        # Show the plot
        plt.show()


if __name__ == '__main__':
    autoencoder = TestAutoEncoder()
    test_img, rec_img, ssim_residual_map, l1_residual_map = autoencoder.get_residual_map()

