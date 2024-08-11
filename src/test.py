import cv2
import gc
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import torch

from skimage import morphology
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

from config.data_paths import JSON_FILES_PATHS
from config.network_config import network_configs
from config.dataset_config import dataset_data_path_selector, dataset_images_path_selector
from models.network_selector import NetworkFactory
from utils.utils import (setup_logger, device_selector, get_patch, patch2img, set_img_color, avg_of_list,
                         find_latest_file_in_latest_directory, create_save_dirs, create_timestamp, load_config_json,
                         file_reader, save_list_to_json)


class TestAutoEncoder:
    def __init__(self):
        timestamp = create_timestamp()
        self.logger = setup_logger()

        train_cfg = (
            load_config_json(
                json_schema_filename=JSON_FILES_PATHS.get_data_path("config_schema_training"),
                json_filename=JSON_FILES_PATHS.get_data_path("config_training")
            )
        )

        self.test_cfg = (
            load_config_json(
                json_schema_filename=JSON_FILES_PATHS.get_data_path("config_schema_testing"),
                json_filename=JSON_FILES_PATHS.get_data_path("config_testing")
            )
        )

        self.network_type = self.test_cfg.get("network_type")
        network_cfg = network_configs(train_cfg).get(self.network_type)
        self.dataset_type = self.test_cfg.get("dataset_type")
        subtest_folder = self.test_cfg.get("subtest_folder")
        self.grayscale = self.test_cfg.get("grayscale")

        self.mask_size = self.test_cfg.get("patch_size") \
            if self.test_cfg.get("img_size")[0] - self.test_cfg.get("crop_size")[0] < self.test_cfg.get("stride") \
            else self.test_cfg.get("img_size")[0]

        # Load paths
        train_dataset_path = (
            dataset_images_path_selector().get(self.dataset_type).get("train")
        )
        self.train_images = (
            file_reader(train_dataset_path, "png")
        )

        test_path = (
            dataset_images_path_selector().get(self.dataset_type).get("test").get(subtest_folder)
        )
        test_images_path = (
            test_path.get("test_images")
        )
        self.test_images = (
            file_reader(test_images_path, "png")
        )

        gt_images_path = (
            test_path.get("ground_truth")
        )
        self.gt_images = (
            file_reader(gt_images_path, "png")
        )

        self.cached_gt_images = (
            self.ground_truth_caching()
        )

        roc_dir = (
            dataset_data_path_selector().get(self.dataset_type).get("roc_plot")
        )
        self.save_roc_plot_dir = (
            create_save_dirs(
                directory_path=roc_dir,
                network_type=self.network_type,
                timestamp=timestamp
            )
        )

        if self.test_cfg.get("vis_results"):
            rec_dir = (
                dataset_data_path_selector().get(self.dataset_type).get("reconstruction_images")
            )
            self.save_reconstruction_plot_dir = (
                create_save_dirs(
                    directory_path=rec_dir,
                    network_type=self.network_type,
                    timestamp=timestamp
                )
            )

        metrics_dir = (
            dataset_data_path_selector().get(self.dataset_type).get("metrics")
        )
        self.metrics_save_dir = (
            create_save_dirs(
                directory_path=metrics_dir,
                network_type=self.network_type,
                timestamp=timestamp
            )
        )

        # Select device to use
        self.device = (
            device_selector(self.test_cfg.get("device"))
        )

        self.model = self.load_model(network_cfg)

    def load_model(self, network_cfg):
        """

        Args:
            network_cfg:

        Returns:

        """

        # Load the model and the latest weights
        model = (
            NetworkFactory.create_network(
                network_type=self.network_type,
                network_cfg=network_cfg)
        ).to(self.device)

        state_dict = torch.load(
            find_latest_file_in_latest_directory(
                path=str(os.path.join(
                    dataset_data_path_selector().get(self.dataset_type).get("model_weights_dir"),
                    self.network_type)
                )
            )
        )

        model.load_state_dict(state_dict)
        model.eval()

        return model

    def get_residual_map(self, test_img_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get the residual map of a test image after reconstruction.
        
        Args:
            test_img_path: Path to the test image.
        
        Returns:
             Tuple containing the original image, reconstructed image, and SSIM residual map.
        """

        if self.grayscale:
            test_img = cv2.imread(test_img_path, cv2.IMREAD_GRAYSCALE)
        else:
            test_img = cv2.imread(test_img_path)

        if test_img.shape[:2] != self.test_cfg.get("img_size"):
            test_img = cv2.resize(test_img, self.test_cfg.get("img_size"))
        if self.test_cfg.get("img_size")[0] != self.mask_size:
            tmp = (self.test_cfg.get("img_size")[0] - self.mask_size) // 2
            test_img = test_img[tmp:tmp + self.mask_size, tmp:tmp + self.mask_size]

        test_img_ = test_img / 255.

        if test_img.shape[:2] == self.test_cfg.get("crop_size"):
            test_img_ = np.expand_dims(test_img_, 0)
            decoded_img = self.model(test_img_)
        else:
            patches = get_patch(test_img_, self.test_cfg.get("crop_size")[0], self.test_cfg.get("stride"))
            if self.grayscale:
                patches = np.expand_dims(patches, 0)
                patches = np.transpose(patches, (1, 0, 2, 3))
            else:
                patches = np.transpose(patches, (0, 3, 1, 2))
            patches = torch.from_numpy(patches).float()
            patches = patches.to(self.device)
            patches = self.model(patches)
            decoded_img = (
                patch2img(
                    patches,
                    self.test_cfg.get("img_size")[0],
                    self.test_cfg.get("crop_size")[0],
                    self.test_cfg.get("stride")
                )
            )

        rec_img = np.reshape((decoded_img * 255.).astype('uint8'), test_img.shape)

        if self.grayscale:
            ssim_residual_map = 1 - ssim(test_img, rec_img, win_size=11, full=True)[1]
        else:
            ssim_residual_map = ssim(test_img, rec_img, win_size=11, full=True, multichannel=True, channel_axis=2)[1]
            ssim_residual_map = 1 - np.mean(ssim_residual_map, axis=2)

        return test_img, rec_img, ssim_residual_map

    def get_mask(self) -> np.ndarray:
        """
        Generate a depressing mask.

        Returns:
             Depressing mask as a NumPy array.
        """

        depr_mask = np.ones((self.mask_size, self.mask_size)) * 0.2
        depr_mask[5:self.mask_size - 5, 5:self.mask_size - 5] = 1
        return depr_mask

    @staticmethod
    def threshold_calculator(start: float, end: float, number_of_steps: int) -> np.ndarray:
        """
        Generate an array of thresholds within a specified range.
        
        Args:
            start: Starting value of the threshold range.
            end: Ending value of the threshold range.
            number_of_steps: Number of steps to divide the range into.

        Returns:
             NumPy array containing the generated thresholds.
        """

        step = end / number_of_steps
        return np.arange(start=start, stop=end, step=step)

    def plot_rec_images(self, test_img: np.ndarray, rec_img: np.ndarray, mask: np.ndarray, vis_img: np.ndarray,
                        idx: int, ssim_threshold: float) -> None:
        """
        Plot and save a grid of images including the test image, reconstructed image, mask, and visualized image.

        Args:
            test_img: The original test image (BGR format).
            rec_img: The reconstructed image (BGR format).
            mask: The mask image (grayscale).
            vis_img: The visualized image (BGR format).
            idx: Index for naming the saved file.
            ssim_threshold:

        Returns:
             None
        """

        filename = os.path.join(self.save_reconstruction_plot_dir, f"{ssim_threshold}_{idx}_reconstruction.png")

        test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)

        plt.subplot(2, 2, 1)
        plt.imshow(test_img, cmap='gray')
        plt.title('test_img')

        plt.subplot(2, 2, 2)
        plt.imshow(rec_img, cmap='gray')
        plt.title('rec_img')

        plt.subplot(2, 2, 3)
        plt.imshow(mask, cmap='gray')
        plt.title('mask')

        plt.subplot(2, 2, 4)
        plt.imshow(vis_img, cmap='gray')
        plt.tight_layout()
        plt.title('vis_img')

        plt.savefig(filename)
        plt.close()

        gc.collect()

    def plot_average_roc(self, all_fpr: list, all_tpr: list) -> float:
        """
        Plot the average ROC curve.

        Args:
            all_fpr: List of false positive rates.
            all_tpr: List of true positive rates.

        Returns:
             auc_roc
        """

        filename = os.path.join(self.save_roc_plot_dir, f"{self.network_type}_{self.dataset_type}roc.png")

        tpr_array = np.array(all_tpr)
        fpr_array = np.array(all_fpr)

        sorted_indices = np.argsort(fpr_array)
        sorted_fpr = fpr_array[sorted_indices]
        sorted_tpr = tpr_array[sorted_indices]

        auc_roc = np.trapz(sorted_tpr, sorted_fpr)

        logging.info(f"ROC AUC: {auc_roc:.4f}")

        plt.plot(sorted_fpr, sorted_tpr, label=f'ROC Curve (AUC = {auc_roc:.4f})')
        plt.scatter(sorted_fpr, sorted_tpr, c='blue', marker='.')
        plt.grid(True)
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend()
        plt.savefig(filename, dpi=300)
        plt.close()

        return auc_roc

    def ground_truth_caching(self) -> dict:
        """

        Returns:

        """

        gt_images_cache = {}
        for gt_img in self.gt_images:
            gt = cv2.imread(gt_img, 0)
            gt = cv2.resize(gt, self.test_cfg.get("img_size"))
            gt = cv2.threshold(gt, 128, 255, cv2.THRESH_BINARY)[1]
            gt = gt.ravel()
            gt = np.where(gt == 255, 1, 0)
            gt_images_cache[gt_img] = gt.ravel()

        return gt_images_cache

    def get_results(self, ssim_threshold: float):
        """
        Calculate average False Positive Rate (FPR) and True Positive Rate (TPR) for a given SSIM threshold.

        Args:
            ssim_threshold: SSIM threshold for generating binary masks.

        Returns:

        """

        all_fpr = []
        all_tpr = []

        for idx, (test_img, gt_img) in enumerate(zip(self.test_images, self.gt_images)):
            test_img, rec_img, ssim_residual_map = self.get_residual_map(test_img)
            depr_mask = self.get_mask()
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

            gt = self.cached_gt_images.get(gt_img)

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

            if self.test_cfg.get("vis_results"):
                vis_img = set_img_color(test_img.copy(), mask_copy, weight_foreground=0.3, grayscale=self.grayscale)
                self.plot_rec_images(test_img, rec_img, mask_copy, vis_img, idx, ssim_threshold)

        return avg_of_list(all_fpr), avg_of_list(all_tpr)

    def calculate_ssim_mse(self):
        ssim_list = []
        mse_list = []

        for idx, train_img in tqdm(
                enumerate(self.train_images),
                total=len(self.train_images),
                desc='Calculating SSIM and MSE'
        ):
            train_img, rec_img, _ = self.get_residual_map(train_img)

            if self.grayscale:
                ssim_res = ssim(train_img, rec_img, win_size=11, full=True)[0]
                mse_res = mean_squared_error(train_img, rec_img)
            else:
                ssim_res = ssim(train_img, rec_img, win_size=11, full=True, multichannel=True, channel_axis=2)[0]
                mse_res = mean_squared_error(train_img.flatten(), rec_img.flatten())

            ssim_list.append(ssim_res)
            mse_list.append(mse_res)

        avg_ssim = avg_of_list(ssim_list)
        mse_avg = avg_of_list(mse_list)

        logging.info(f"SSIM: {avg_ssim:.4f}")
        logging.info(f"MSE: {mse_avg:.4f}")

        return avg_ssim, mse_avg

    def main(self) -> None:
        """
        Main method for executing the ROC analysis.

        Returns:
             None
        """

        threshold_range = (
            self.threshold_calculator(
                start=self.test_cfg.get("threshold_init"),
                end=self.test_cfg.get("threshold_end"),
                number_of_steps=self.test_cfg.get("num_of_steps")
            )
        )

        fpr_list, tpr_list = [], []

        for ssim_tresh in tqdm(
                threshold_range,
                total=len(threshold_range),
                desc='Calculating FPR and TPR'
        ):
            fpr, tpr = self.get_results(ssim_tresh)
            fpr_list.append(fpr)
            tpr_list.append(tpr)

        filename = os.path.join(self.metrics_save_dir, 'fpr_tpr_ssim_mse_roc_auc_results.json')
        avg_ssim, mse_avg = self.calculate_ssim_mse()
        auc_roc = self.plot_average_roc(fpr_list, tpr_list)

        results = {
            "fpr_tpr": fpr_list,
            "tpr_tpr": tpr_list,
            "avg_ssim": avg_ssim,
            "mse_avg": mse_avg,
            "auc_roc": auc_roc,
        }

        save_list_to_json(filename=filename, results_dict=results)


if __name__ == '__main__':
    try:
        autoencoder = TestAutoEncoder()
        autoencoder.main()
    except KeyboardInterrupt as kie:
        logging.error(kie)
