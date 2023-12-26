import logging

import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import torch

from glob import glob
from skimage import morphology
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import confusion_matrix, roc_curve, auc

from config.config import ConfigTesting
from config.network_config import network_configs, dataset_data_path_selector
from models.network_selector import NetworkFactory
from utils.utils import (setup_logger, use_gpu_if_available, get_patch, patch2img, set_img_color,
                         find_latest_file_in_latest_directory)


class TestAutoEncoder:
    def __init__(self):
        self.logger = setup_logger()
        self.test_cfg = ConfigTesting().parse()
        network_cfg = network_configs().get(self.test_cfg.network_type)

        self.mask_size = self.test_cfg.patch_size \
            if self.test_cfg.img_size[0] - self.test_cfg.crop_size[0] < self.test_cfg.stride \
            else self.test_cfg.img_size[0]

        self.device = use_gpu_if_available()
        self.model = NetworkFactory.create_network(network_type=self.test_cfg.network_type,
                                                   network_cfg=network_cfg,
                                                   device=self.device)

        state_dict = torch.load(
            find_latest_file_in_latest_directory(
                path=str(os.path.join(
                    dataset_data_path_selector().get(self.test_cfg.dataset_type).get("model_weights_dir"),
                    self.test_cfg.network_type)
                )
            )
        )

        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def get_residual_map(self, test_img_path):
        test_img = cv2.imread(test_img_path)

        if test_img.shape[:2] != self.test_cfg.img_size:
            test_img = cv2.resize(test_img, self.test_cfg.img_size)
        if self.test_cfg.img_size[0] != self.mask_size:
            tmp = (self.test_cfg.img_size[0] - self.mask_size) // 2
            test_img = test_img[tmp:tmp + self.mask_size, tmp:tmp + self.mask_size]

        test_img_ = test_img / 255.

        if test_img.shape[:2] == self.test_cfg.crop_size:
            test_img_ = np.expand_dims(test_img_, 0)
            decoded_img = self.model(test_img_)
        else:
            patches = get_patch(test_img_, self.test_cfg.crop_size[0], self.test_cfg.stride)
            patches = np.transpose(patches, (0, 3, 1, 2))
            patches = torch.from_numpy(patches).float()
            patches = patches.to(self.device)
            patches = self.model(patches)
            decoded_img = (
                patch2img(patches, self.test_cfg.img_size[0], self.test_cfg.crop_size[0], self.test_cfg.stride)
            )

        rec_img = np.reshape((decoded_img * 255.).astype('uint8'), test_img.shape)
        _, ssim_residual_map = ssim(test_img, rec_img, full=True, multichannel=True, channel_axis=2)
        ssim_residual_map = 1 - np.mean(ssim_residual_map, axis=2)

        return test_img, rec_img, ssim_residual_map

    def get_depressing_mask(self):
        depr_mask = np.ones((self.mask_size, self.mask_size)) * 0.2
        depr_mask[5:self.mask_size - 5, 5:self.mask_size - 5] = 1
        return depr_mask

    @staticmethod
    def plot_results(test_img, rec_img, mask, vis_img):
        test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)

        plt.subplot(2, 2, 1)
        plt.imshow(test_img)
        plt.title('test_img')

        plt.subplot(2, 2, 2)
        plt.imshow(rec_img)
        plt.title('rec_img')

        plt.subplot(2, 2, 3)
        plt.imshow(mask, cmap='gray')
        plt.title('mask')

        plt.subplot(2, 2, 4)
        plt.imshow(vis_img, cmap='gray')
        plt.title('vis_img')

        plt.tight_layout()
        plt.show()

    def get_results(self, ssim_threshold):
        images = sorted(glob('D:/mvtec/bottle/test/good' + "/*.png"))

        for img in images:
            test_img, rec_img, ssim_residual_map = self.get_residual_map(img)
            depr_mask = self.get_depressing_mask()
            ssim_residual_map *= depr_mask

            mask = np.zeros((self.mask_size, self.mask_size))
            mask[ssim_residual_map > ssim_threshold] = 1

            kernel = morphology.disk(4)
            mask = morphology.opening(mask, kernel)
            mask *= 255
            mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)[1]

            gt = np.zeros((mask.shape[0], mask.shape[1]))  # Assuming anomalies in the ground truth are represented by 1
            mask = mask.ravel()
            gt = gt.ravel()

            CM = confusion_matrix(gt, mask)

            # Extract True Negative (TN), False Positive (FP), False Negative (FN), True Positive (TP)
            TN, FP, FN, TP = CM.ravel()

            fpr, tpr, _ = roc_curve(gt, mask)
            tpr = TP / (TP + FN) if (TP + FN) != 0 else 0
            roc_auc = auc(fpr, tpr)

            print('ROC AUC', roc_auc)
            # vis_img = set_img_color(test_img.copy(), mask, weight_foreground=0.3)
            #
            # self.plot_results(test_img, rec_img, mask, vis_img)


if __name__ == '__main__':
    try:
        autoencoder = TestAutoEncoder()
        # for i in np.arange(0, 1.1, 0.1):
        autoencoder.get_results(0.4)
    except KeyboardInterrupt as kie:
        logging.error(kie)
