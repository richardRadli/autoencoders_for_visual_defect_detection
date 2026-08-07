import cv2
import gc
import json
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

from services.defect_detection.app.core.utility_services.config_service.architecture_config_service import ArchitectureConfigService
from services.defect_detection.app.core.models.network_selector import NetworkFactory
from shared.core.path_bindings import config_paths, dataset_paths, training_testing_paths
from utils.ml_utils import device_selector, patch2img
from utils.system_utils import (setup_logger, get_patch, set_img_color, avg_of_list,
                                find_latest_file_in_latest_directory, create_save_dirs, create_timestamp,
                                file_reader, save_list_to_json)


class TestAutoEncoder:
    def __init__(self, config: dict):
        """
        Set up the model, data and output dirs for one evaluation run.

        img_size and crop_size are not chosen here: they are inherited from the
        training run of the selected weights (which took them from the
        augmentation run), so the architecture always matches the weights.

        Args:
            config: Effective testing config (already resolved by the API).
        """
        self.timestamp = create_timestamp()
        setup_logger()

        self.test_cfg = config

        self._progress_callback = None
        self._current_phase = None
        self._phase_start = 0.0
        self._phase_span = 0.0
        self._phase_total = 0
        self._phase_done = 0
        self._last_emitted = None

        self.network_type = self.test_cfg.get("network_type")
        self.dataset_type = self.test_cfg.get("dataset_type")
        self.subtest_folder = self.test_cfg.get("subtest_folder")

        self.stride = self.test_cfg.get("stride")
        self.weights_run = self.test_cfg.get("weights_run")

        if self.network_type not in ["AE", "AEE", "DAE", "DAEE"]:
            raise ValueError(f"wrong network type: {self.network_type}")

        self.device = device_selector(preferred_device="cuda")

        weights_root = os.path.join(
            str(training_testing_paths(self.dataset_type)["model_weights"]), self.network_type
        )
        if not os.path.isdir(weights_root):
            raise ValueError(f"No trained weights found for {self.network_type} / {self.dataset_type}")
        self.weights_path = self._select_weights_path(weights_root)

        train_params = self.load_train_params(os.path.dirname(self.weights_path))
        self.grayscale = train_params["grayscale"]

        # Sizes are inherited from the training run of these weights.
        self.img_size, self.crop_size = self._read_train_sizes(train_params)

        self.mask_size = self.img_size
        self.depr_mask = self.get_mask()

        network_cfg = ArchitectureConfigService.build(
            base_path=config_paths().get("network_config"),
            network_type=self.network_type,
            grayscale=self.grayscale,
            latent_space_dimension=train_params["latent_space_dimension"],
            crop_size=self.crop_size,
        )

        self.model = self.load_model(network_cfg)

        paths = dataset_paths(self.dataset_type)

        if self.subtest_folder not in paths["test"]:
            raise ValueError(
                f"Invalid subtest_folder '{self.subtest_folder}' for dataset '{self.dataset_type}'"
            )

        test_images_path = str(paths["test"][self.subtest_folder])
        self.test_images = file_reader(test_images_path, "png", "jpg")
        if not self.test_images:
            raise ValueError(f"No test images found in {test_images_path}")

        if not self.test_cfg.get("vis_reconstruction"):
            self.train_images = file_reader(file_path=str(paths["good"]), extension="png", extension2="jpg")
            if not self.train_images:
                raise ValueError(f"No training images found in {paths['good']}")

            gt_images_path = str(paths["gt"][self.subtest_folder])
            self.gt_images = file_reader(gt_images_path, "png", "jpg")
            if len(self.test_images) != len(self.gt_images):
                raise ValueError(
                    f"test_images count ({len(self.test_images)}) != ground_truth count ({len(self.gt_images)})"
                )

            self.cached_gt_images = self.ground_truth_caching()

            tt_paths = training_testing_paths(self.dataset_type)
            self.save_roc_plot_dir = create_save_dirs(
                directory_path=str(tt_paths["roc_plot"]),
                network_type=self.network_type,
                timestamp=self.timestamp,
            )
            self.metrics_save_dir = create_save_dirs(
                directory_path=str(tt_paths["metrics"]),
                network_type=self.network_type,
                timestamp=self.timestamp,
            )
            if self.test_cfg.get("vis_results"):
                self.save_reconstruction_plot_dir = create_save_dirs(
                    directory_path=str(tt_paths["reconstruction_vis"]),
                    network_type=self.network_type,
                    timestamp=self.timestamp,
                )
        else:
            self.save_reconstruction_dir = create_save_dirs(
                directory_path=str(training_testing_paths(self.dataset_type)["reconstruction"]),
                network_type=self.network_type,
                timestamp=self.timestamp,
            )

    def _select_weights_path(self, weights_root: str) -> str:
        """
        Resolve the weight file to load for this evaluation run.

        Without a selected run the latest run's weight is used (unchanged
        default behavior). With a selected weights_run, only a direct timestamp
        subfolder of this dataset+network weights root is accepted (guarding
        against path traversal), and its highest-epoch .pt is loaded.

        Args:
            weights_root: The dataset+network model_weights directory.

        Returns:
            str: Absolute path to the selected weight file.

        Raises:
            ValueError: If no weights exist, or the selected run is unknown or
                unsafe, or it holds no weight file.
        """
        if not self.weights_run:
            try:
                return find_latest_file_in_latest_directory(path=weights_root, extension=".pt")
            except ValueError:
                raise ValueError(
                    f"No trained weights found for {self.network_type} / {self.dataset_type}"
                )

        root_resolved = os.path.realpath(weights_root)
        run_dir = os.path.realpath(os.path.join(root_resolved, self.weights_run))

        if os.path.dirname(run_dir) != root_resolved or not os.path.isdir(run_dir):
            raise ValueError(
                f"Unknown weights_run '{self.weights_run}' for "
                f"{self.network_type} / {self.dataset_type}"
            )

        weight_files = file_reader(run_dir, "pt")
        if not weight_files:
            raise ValueError(
                f"No weight file in weights_run '{self.weights_run}' for "
                f"{self.network_type} / {self.dataset_type}"
            )

        # file_reader sorts numerically, so the last file is the highest epoch.
        return weight_files[-1]

    def _init_progress(self, callback) -> None:
        """
        Reset the progress state and store the reporting callback.

        Progress is normalized to a 0-100 percentage so the frontend bar can
        render it directly (current = percent, total = 100).

        Args:
            callback: Callable(percent, total, phase) that reports one update,
                or None to disable progress reporting.
        """
        self._progress_callback = callback
        self._current_phase = None
        self._phase_start = 0.0
        self._phase_span = 0.0
        self._phase_total = 0
        self._phase_done = 0
        self._last_emitted = None

    def _begin_phase(self, phase: str, start: float, span: float, total: int) -> None:
        """
        Start a progress phase that maps its items onto [start, start + span] %.

        The phase-start percentage is emitted immediately (with the new phase
        label), which also serves as the run's initial 0% for the first phase.

        Args:
            phase: Short phase label (residual_maps / thresholds / metrics /
                reconstruction).
            start: The global percentage at which this phase begins.
            span: The global percentage width this phase covers.
            total: Number of items processed during this phase.
        """
        self._current_phase = phase
        self._phase_start = start
        self._phase_span = span
        self._phase_total = max(total, 0)
        self._phase_done = 0
        self._emit(force=True)

    def _advance(self, step: int = 1) -> None:
        """
        Mark items done in the current phase and emit if the percentage moved.

        Args:
            step: How many items were just processed.
        """
        self._phase_done += step
        self._emit()

    def _emit(self, force: bool = False) -> None:
        """
        Report the current percentage, throttled to whole-percent changes.

        Always emits on a phase change (force) and on every new integer percent;
        the final 100% is emitted naturally as the last phase completes.

        Args:
            force: Emit even if the percentage did not increase (phase change).
        """
        if self._progress_callback is None:
            return

        if self._phase_total > 0:
            fraction = min(self._phase_done / self._phase_total, 1.0)
        else:
            fraction = 1.0

        percent = int(round(self._phase_start + fraction * self._phase_span))
        percent = max(0, min(100, percent))

        if not force and self._last_emitted is not None and percent <= self._last_emitted:
            return

        self._progress_callback(percent, 100, self._current_phase)
        self._last_emitted = percent

    @staticmethod
    def load_train_params(weights_dir: str) -> dict:
        """
        Load the params.json saved next to the trained weights.

        Args:
            weights_dir: Directory of the selected weight file.

        Returns:
            dict: The training run parameters.
        """
        params_path = os.path.join(weights_dir, "params.json")
        if not os.path.isfile(params_path):
            raise ValueError(f"params.json not found in {weights_dir} - run a new training to generate it")

        with open(params_path, "r", encoding="utf-8") as f:
            params = json.load(f)

        logging.info(f"Loaded training params from: {params_path}")
        return params

    @staticmethod
    def _read_train_sizes(train_params: dict) -> tuple[int, int]:
        """
        Read the (img_size, crop_size) the weights were trained with.

        Both sizes are inherited from training, which took them from the
        augmentation run; they are never chosen at test time.

        Args:
            train_params: The training params.json loaded next to the weights.

        Returns:
            tuple[int, int]: (img_size, crop_size).

        Raises:
            ValueError: If either size is missing or is not a positive integer.
        """
        img_size = train_params.get("img_size")
        crop_size = train_params.get("crop_size")

        for name, value in (("img_size", img_size), ("crop_size", crop_size)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(
                    f"Trained weights have no usable {name} in params.json (got {value!r}) - "
                    f"retrain so the sizes are recorded"
                )

        return img_size, crop_size

    def load_model(self, network_cfg: dict):
        """
        Build the network and load the latest trained weights.

        Args:
            network_cfg: The architecture config for NetworkFactory.

        Returns:
            torch.nn.Module: The model in eval mode on the selected device.
        """
        model = NetworkFactory.create_network(
            network_type=self.network_type, network_cfg=network_cfg
        ).to(self.device)

        state_dict = torch.load(self.weights_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.eval()

        logging.info(f"Loaded weights from: {self.weights_path}")
        return model

    def get_residual_map(self, test_img_path: str) -> tuple:
        """
        Get the residual map of a test image after reconstruction.

        Args:
            test_img_path: Path to the test image.

        Returns:
            tuple: The original image, reconstructed image and SSIM residual map.
        """
        if self.grayscale:
            test_img = cv2.imread(test_img_path, cv2.IMREAD_GRAYSCALE)
        else:
            test_img = cv2.imread(test_img_path)

        if test_img is None:
            raise ValueError(f"Failed to read image: {test_img_path}")

        if not self.grayscale:
            test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

        if test_img.shape[:2] != (self.img_size, self.img_size):
            test_img = cv2.resize(test_img, (self.img_size, self.img_size))

        test_img_ = test_img / 255.

        patches = get_patch(test_img_, self.crop_size, self.stride)
        if self.grayscale:
            patches = np.expand_dims(patches, 0)
            patches = np.transpose(patches, (1, 0, 2, 3))
        else:
            patches = np.transpose(patches, (0, 3, 1, 2))
        patches = torch.from_numpy(patches).float().to(self.device)

        with torch.no_grad():
            patches = self.model(patches)

        decoded_img = patch2img(patches, self.img_size, self.crop_size, self.stride)
        rec_img = np.reshape((decoded_img * 255.).astype('uint8'), test_img.shape)

        if self.grayscale:
            ssim_residual_map = 1 - ssim(test_img, rec_img, win_size=11, full=True)[1]
        else:
            ssim_residual_map = ssim(test_img, rec_img, win_size=11, full=True, multichannel=True, channel_axis=2)[1]
            ssim_residual_map = 1 - np.mean(ssim_residual_map, axis=2)

        return test_img, rec_img, ssim_residual_map

    def get_mask(self) -> np.ndarray:
        """
        Generate a depressing mask that damps the image borders.

        Returns:
            np.ndarray: The depressing mask.
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
            np.ndarray: The generated thresholds.
        """
        step = (end - start) / number_of_steps
        return np.arange(start=start, stop=end, step=step)

    def plot_ori_rec_images(self) -> None:
        """
        Save side-by-side original and reconstructed images of the test set.

        Returns:
            None
        """
        for idx, test_img_path in tqdm(enumerate(self.test_images), total=len(self.test_images), desc="Reconstructing"):
            filename = os.path.join(str(self.save_reconstruction_dir), f"{idx}_reconstruction.png")
            test_img, rec_img, _ = self.get_residual_map(test_img_path)

            plt.subplot(1, 2, 1)
            plt.imshow(test_img, cmap='gray')
            plt.title('test_img')

            plt.subplot(1, 2, 2)
            plt.imshow(rec_img, cmap='gray')
            plt.title('rec_img')

            plt.tight_layout()
            plt.savefig(filename, dpi=300)
            plt.close()
            gc.collect()

            self._advance()

    def plot_ori_rec_mask_images(self, test_img: np.ndarray, rec_img: np.ndarray, mask: np.ndarray,
                                 vis_img: np.ndarray, idx: int, ssim_threshold: float) -> None:
        """
        Plot and save the test image, reconstruction, mask and visualization.

        Args:
            test_img: The original test image.
            rec_img: The reconstructed image.
            mask: The predicted mask.
            vis_img: The visualization image.
            idx: Index for naming the saved file.
            ssim_threshold: The threshold used for the mask.

        Returns:
            None
        """

        threshold_str = f"{ssim_threshold:.3f}".replace(".", "")
        filename = os.path.join(str(self.save_reconstruction_plot_dir), f"{idx}_{threshold_str}_reconstruction.png")

        if test_img.ndim == 3:
            test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
        if vis_img.ndim == 3:
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

        plt.savefig(filename, dpi=300)
        plt.close()
        gc.collect()

    def plot_average_roc(self, all_fpr: list, all_tpr: list) -> float:
        """
        Plot the average ROC curve and return its AUC.

        Args:
            all_fpr: List of false positive rates.
            all_tpr: List of true positive rates.

        Returns:
            float: The ROC AUC value.
        """
        filename = os.path.join(str(self.save_roc_plot_dir), f"{self.network_type}_{self.dataset_type}_roc.png")

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

        return float(auc_roc)

    def ground_truth_caching(self) -> dict:
        """
        Load, binarize and cache all ground truth masks.

        Returns:
            dict: Ground truth image path -> flattened binary mask.
        """
        gt_images_cache = {}
        for gt_img in self.gt_images:
            gt = cv2.imread(gt_img, 0)
            if gt is None:
                raise ValueError(f"Failed to read ground truth image: {gt_img}")
            gt = cv2.resize(gt, (self.img_size, self.img_size))
            gt = cv2.threshold(gt, 128, 255, cv2.THRESH_BINARY)[1]
            gt = gt.ravel()
            gt = np.where(gt == 255, 1, 0)
            gt_images_cache[gt_img] = gt.ravel()

        return gt_images_cache

    def build_residual_cache(self) -> list:
        """
        Compute and cache the reconstruction and residual map of every test
        image once, so the threshold sweep does not re-run the model.

        Returns:
            list: One (test_img, rec_img, ssim_residual_map, gt) tuple per image.
        """
        cache = []
        for test_img_path, gt_img_path in tqdm(
            zip(self.test_images, self.gt_images),
            total=len(self.test_images),
            desc="Computing residual maps",
        ):
            test_img, rec_img, ssim_residual_map = self.get_residual_map(test_img_path)
            gt = self.cached_gt_images.get(gt_img_path)
            cache.append((test_img, rec_img, ssim_residual_map, gt))
            self._advance()
        return cache

    def get_results(self, ssim_threshold: float, save_vis: bool) -> tuple:
        """
        Calculate average FPR and TPR at a threshold from the cached residuals.

        Args:
            ssim_threshold: SSIM threshold for generating binary masks.
            save_vis: Whether to save per-image visualizations at this threshold.

        Returns:
            tuple: The average FPR and TPR.
        """
        all_fpr = []
        all_tpr = []

        for idx, (test_img, rec_img, ssim_residual_map, gt) in enumerate(self.residual_cache):
            residual = ssim_residual_map * self.depr_mask

            mask = np.zeros((self.mask_size, self.mask_size))
            mask[residual > ssim_threshold] = 1

            kernel = morphology.disk(4)
            mask = morphology.opening(mask, kernel)
            mask *= 255
            mask_copy = mask
            mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)[1]
            mask = np.uint8(mask.ravel())
            mask = np.where(mask == 255, 1, 0)

            conf_mtx = confusion_matrix(gt, mask, labels=[0, 1])

            true_neg = conf_mtx[0][0]
            false_pos = conf_mtx[0][1]
            false_neg = conf_mtx[1][0]
            true_pos = conf_mtx[1][1]

            false_pos_rate = 0 if false_pos + true_neg == 0 else false_pos / (false_pos + true_neg)
            all_fpr.append(false_pos_rate)

            true_pos_rate = 0
            if true_pos != 0 or false_neg != 0:
                true_pos_rate = true_pos / (true_pos + false_neg)
            all_tpr.append(true_pos_rate)

            if save_vis:
                vis_img = set_img_color(test_img.copy(), mask_copy, weight_foreground=0.3, grayscale=self.grayscale)
                self.plot_ori_rec_mask_images(test_img, rec_img, mask_copy, vis_img, idx, ssim_threshold)

        return avg_of_list(all_fpr), avg_of_list(all_tpr)

    def calculate_ssim_mse(self) -> tuple:
        """
        Calculate the average SSIM and MSE over the train images.

        Returns:
            tuple: The average SSIM and MSE.
        """
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

            self._advance()

        avg_ssim = avg_of_list(ssim_list)
        mse_avg = avg_of_list(mse_list)

        logging.info(f"SSIM: {avg_ssim:.4f}")
        logging.info(f"MSE: {mse_avg:.4f}")

        return avg_ssim, mse_avg

    def run(self, progress_callback=None) -> dict:
        """
        Execute the ROC/SSIM/MSE evaluation or the reconstruction visualization.

        Args:
            progress_callback: Optional callable(percent, total, phase) invoked as
                items are processed so the caller (the Celery task) can report
                evaluation progress. None disables progress reporting.

        Returns:
            dict: Summary of the run with the main metrics and output paths.
        """
        self._init_progress(progress_callback)

        if not self.test_cfg.get("vis_reconstruction"):
            threshold_range = self.threshold_calculator(
                start=self.test_cfg.get("threshold_init"),
                end=self.test_cfg.get("threshold_end"),
                number_of_steps=self.test_cfg.get("num_of_steps"),
            )

            if len(threshold_range) == 0:
                raise ValueError("Empty threshold range - check threshold_init/threshold_end/num_of_steps")

            # The three phases each span one third of the bar: residual maps
            # (0-33%), the threshold sweep (33-66%) and the SSIM/MSE pass
            # (66-100%). Progress is reported as a normalized 0-100 percentage.
            third = 100.0 / 3.0

            self._begin_phase("residual_maps", 0.0, third, len(self.test_images))
            self.residual_cache = self.build_residual_cache()

            vis_interval = self.test_cfg.get("vis_interval")
            do_vis = bool(self.test_cfg.get("vis_results"))
            fpr_list, tpr_list = [], []

            self._begin_phase("thresholds", third, third, len(threshold_range))
            for t_idx, ssim_tresh in enumerate(
                tqdm(threshold_range, total=len(threshold_range), desc='Calculating FPR and TPR')
            ):
                save_vis = do_vis and (t_idx % vis_interval == 0)
                fpr, tpr = self.get_results(ssim_tresh, save_vis)
                fpr_list.append(float(fpr))
                tpr_list.append(float(tpr))
                self._advance()

            filename = os.path.join(str(self.metrics_save_dir), "fpr_tpr_ssim_mse_roc_auc_results.json")

            self._begin_phase("metrics", 2.0 * third, third, len(self.train_images))
            avg_ssim, mse_avg = self.calculate_ssim_mse()
            auc_roc = self.plot_average_roc(fpr_list, tpr_list)

            results = {
                "fpr": fpr_list,
                "tpr": tpr_list,
                "avg_ssim": float(avg_ssim),
                "mse_avg": float(mse_avg),
                "auc_roc": auc_roc,
            }
            save_list_to_json(filename=filename, results_dict=results)

            return {
                "status": "DONE",
                "network_type": self.network_type,
                "dataset_type": self.dataset_type,
                "subtest_folder": self.subtest_folder,
                "auc_roc": auc_roc,
                "avg_ssim": float(avg_ssim),
                "mse_avg": float(mse_avg),
                "metrics_path": filename,
                "weights_used": self.weights_path,
            }

        self._begin_phase("reconstruction", 0.0, 100.0, len(self.test_images))
        self.plot_ori_rec_images()
        return {
            "status": "DONE",
            "network_type": self.network_type,
            "dataset_type": self.dataset_type,
            "reconstructed_images": len(self.test_images),
            "save_dir": str(self.save_reconstruction_dir),
            "weights_used": self.weights_path,
        }