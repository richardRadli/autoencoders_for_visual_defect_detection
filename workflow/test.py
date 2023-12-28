import cv2
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import torch

from glob import glob
from skimage import morphology
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from config.config import ConfigTesting
from config.network_config import network_configs, dataset_data_path_selector, dataset_images_path_selector
from models.network_selector import NetworkFactory
from utils.utils import (setup_logger, use_gpu_if_available, get_patch, patch2img, set_img_color, avg_of_list,
                         find_latest_file_in_latest_directory, create_save_dirs, create_timestamp)


class TestAutoEncoder:
    def __init__(self):
        timestamp = create_timestamp()
        self.logger = setup_logger()
        self.test_cfg = ConfigTesting().parse()
        network_cfg = network_configs().get(self.test_cfg.network_type)

        self.mask_size = self.test_cfg.patch_size \
            if self.test_cfg.img_size[0] - self.test_cfg.crop_size[0] < self.test_cfg.stride \
            else self.test_cfg.img_size[0]

        test_dataset_path = (dataset_images_path_selector().get(self.test_cfg.dataset_type).get("test"))
        self.test_images = sorted(glob(os.path.join(test_dataset_path, "*.png")))

        gt_dataset_path = (dataset_images_path_selector().get(self.test_cfg.dataset_type).get("gt"))
        self.gt_images = sorted(glob(os.path.join(gt_dataset_path, "*.png")))

        roc_dir = dataset_data_path_selector().get(self.test_cfg.dataset_type).get("roc_plot")
        self.save_roc_plot_dir = create_save_dirs(directory_path=roc_dir,
                                                  network_type=self.test_cfg.network_type,
                                                  timestamp=timestamp)

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
        """

        :param test_img_path:
        :return:
        """

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
        """

        :return:
        """

        depr_mask = np.ones((self.mask_size, self.mask_size)) * 0.2
        depr_mask[5:self.mask_size - 5, 5:self.mask_size - 5] = 1
        return depr_mask

    @staticmethod
    def plot_results(test_img, rec_img, mask, vis_img):
        """

        :param test_img:
        :param rec_img:
        :param mask:
        :param vis_img:
        :return:
        """

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

    @staticmethod
    def threshold_calculator(start: float, end: float, number_of_steps: float):
        """

        :param start:
        :param end:
        :param number_of_steps:
        :return:
        """

        step = end / number_of_steps
        return np.arange(start=start, stop=end, step=step)

    def plot_average_roc(self, all_fpr, all_tpr):
        """

        :param all_fpr:
        :param all_tpr:
        :return:
        """

        filename = os.path.join(self.save_roc_plot_dir, "roc.png")

        tpr_array = np.array(all_tpr)
        fpr_array = np.array(all_fpr)

        sorted_indices = np.argsort(fpr_array)
        sorted_fpr = fpr_array[sorted_indices]
        sorted_tpr = tpr_array[sorted_indices]

        auc_roc = np.trapz(sorted_tpr, sorted_fpr)

        plt.plot(sorted_fpr, sorted_tpr, label=f'ROC Curve (AUC = {auc_roc:.4f})')
        plt.scatter(sorted_fpr, sorted_tpr, c='blue', marker='.')
        plt.grid(True)
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend()
        plt.savefig(filename)
        plt.close()

    def get_results(self, ssim_threshold):
        """

        :param ssim_threshold:
        :return:
        """

        all_fpr = []
        all_tpr = []

        for idx, (test_img, gt_img) in enumerate(zip(self.test_images, self.gt_images)):
            test_img, rec_img, ssim_residual_map = self.get_residual_map(test_img)
            depr_mask = self.get_depressing_mask()
            ssim_residual_map *= depr_mask

            mask = np.zeros((self.mask_size, self.mask_size))
            mask[ssim_residual_map > ssim_threshold] = 1

            kernel = morphology.disk(4)
            mask = morphology.opening(mask, kernel)
            mask *= 255
            mask_copy = mask
            mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)[1]
            mask = np.uint8(mask.ravel())
            mask = np.where(mask == 255, 1, 0)

            gt = cv2.imread(gt_img, 0)
            gt = cv2.resize(gt, self.test_cfg.img_size)
            gt = cv2.threshold(gt, 128, 255, cv2.THRESH_BINARY)[1]
            gt = gt.ravel()
            gt = np.where(gt == 255, 1, 0)

            conf_mtx = confusion_matrix(gt, mask)

            true_neg = conf_mtx[0][0]
            false_pos = conf_mtx[0][1]
            false_neg = conf_mtx[1][0]
            true_pos = conf_mtx[1][1]

            # FPR
            false_pos_rate = false_pos / (false_pos + true_neg)
            all_fpr.append(false_pos_rate)

            # TPR
            true_pos_rate = 0
            if true_pos != 0 or false_neg != 0:
                true_pos_rate = true_pos / (true_pos + false_neg)
            all_tpr.append(true_pos_rate)

            if self.test_cfg.vis_results:
                vis_img = set_img_color(test_img.copy(), mask_copy, weight_foreground=0.3)
                self.plot_results(test_img, rec_img, mask_copy, vis_img)

        return avg_of_list(all_fpr), avg_of_list(all_tpr)


if __name__ == '__main__':
    try:
        autoencoder = TestAutoEncoder()
        threshold_range = autoencoder.threshold_calculator(start=0.01, end=1.01, number_of_steps=101)

        fpr_list, tpr_list = [], []
        for ssim_tresh in tqdm(threshold_range):
            fpr, tpr = autoencoder.get_results(ssim_tresh)
            fpr_list.append(fpr)
            tpr_list.append(tpr)
        autoencoder.plot_average_roc(fpr_list, tpr_list)
    except KeyboardInterrupt as kie:
        logging.error(kie)
