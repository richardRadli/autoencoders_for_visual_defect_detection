import logging
import os

import numpy as np
import torch
import torch.optim as optim

from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from pytorch_msssim import SSIM
from typing import Tuple

from services.defect_detection.app.core.utility_services.config_service.architecture_config_service import ArchitectureConfigService
from services.defect_detection.app.core.dataloaders.data_loader_ae import MVTecDataset
from services.defect_detection.app.core.dataloaders.data_loader_dae import MVTecDatasetDenoising
from services.defect_detection.app.core.models.network_selector import NetworkFactory
from shared.core.path_bindings import config_paths, dataset_paths, training_testing_paths
from utils.ml_utils import device_selector, set_seed, visualize_images
from utils.system_utils import create_save_dirs, create_timestamp, find_latest_directory, read_json_safely, save_list_to_json, setup_logger


class TrainAutoEncoder:
    def __init__(self, config: dict):
        """
        Set up the model, data, loss, optimizer and save paths for one training run.

        Args:
            config: Effective training config (already resolved by the API).
        """
        self.timestamp = create_timestamp()
        setup_logger()

        self.train_cfg = config

        if self.train_cfg.get("seed"):
            set_seed(seed=1234)

        self.network_type = self.train_cfg.get("network_type")
        self.dataset_type = self.train_cfg.get("dataset_type")

        if self.network_type not in ["AE", "AEE", "DAE", "DAEE"]:
            raise ValueError(f"wrong network type: {self.network_type}")

        # img_size and crop_size come only from the augmentation run the training
        # data was built with, and are recorded so the saved params keep them.
        self.aug_dir = find_latest_directory(str(dataset_paths(self.dataset_type)["aug"]))
        self.img_size, self.crop_size = self._read_aug_sizes(self.aug_dir)
        self.train_cfg = {
            **self.train_cfg,
            "img_size": self.img_size,
            "crop_size": self.crop_size,
        }

        network_cfg = ArchitectureConfigService.build(
            base_path=config_paths().get("network_config"),
            network_type=self.network_type,
            grayscale=self.train_cfg.get("grayscale"),
            latent_space_dimension=self.train_cfg.get("latent_space_dimension"),
            crop_size=self.crop_size,
        )

        self.device = device_selector(preferred_device="cuda")

        self.model = NetworkFactory.create_network(
            network_type=self.network_type, network_cfg=network_cfg
        ).to(self.device)

        self.train_dataloader, self.valid_dataloader = self.create_dataset()

        self.criterion = SSIM(
            win_sigma=1.5, data_range=1, size_average=True,
            channel=1 if self.train_cfg.get("grayscale") else 3,
        )

        self.optimizer = optim.Adam(
            params=self.model.parameters(), lr=self.train_cfg.get("learning_rate")
        )

        self.scheduler = StepLR(
            optimizer=self.optimizer,
            step_size=self.train_cfg.get("step_size"),
            gamma=self.train_cfg.get("gamma"),
        )

        tensorboard_log_dir = create_save_dirs(
            directory_path=str(training_testing_paths(self.dataset_type)["model_logs"]),
            network_type=self.network_type,
            timestamp=self.timestamp,
        )

        self.writer = SummaryWriter(
            log_dir=str(tensorboard_log_dir)
        )

        self.save_path = create_save_dirs(
            directory_path=str(training_testing_paths(self.dataset_type)["model_weights"]),
            network_type=self.network_type,
            timestamp=self.timestamp,
        )

        self.params_path = os.path.join(str(self.save_path), "params.json")
        save_list_to_json(
            filename=self.params_path,
            results_dict={**self.train_cfg, "last_completed_epoch": 0},
        )
        logging.info(f"Saved training params to {self.params_path}")

    @staticmethod
    def _read_aug_sizes(aug_dir: str) -> Tuple[int, int]:
        """
        Read the (img_size, crop_size) the given augmentation run was built with.

        The augmentation run is the single source of both sizes: the model's
        bottleneck kernel is selected from crop_size, and both are recorded in the
        training params so testing can read them back from the weights.

        Args:
            aug_dir: Path to the augmentation run folder.

        Returns:
            Tuple[int, int]: (img_size, crop_size) from the run's params.json.

        Raises:
            ValueError: If params.json is missing/unreadable or lacks img_size/crop_size.
        """
        aug_params = read_json_safely(os.path.join(aug_dir, "params.json"))
        run_params = aug_params.get("params") if isinstance(aug_params, dict) else None
        if not isinstance(run_params, dict) or "img_size" not in run_params or "crop_size" not in run_params:
            raise ValueError(f"Latest augmentation run has no usable img_size/crop_size: {aug_dir}")
        return run_params["img_size"], run_params["crop_size"]

    def create_dataset(self) -> Tuple[DataLoader, DataLoader]:
        """
        Create and split the dataset into train/validation dataloaders.

        Returns:
            tuple: The training and validation DataLoaders.
        """
        paths = dataset_paths(self.dataset_type)
        aug_dir = self.aug_dir

        if self.network_type in ["AE", "AEE"]:
            dataset = MVTecDataset(
                root_dir=aug_dir, grayscale=self.train_cfg.get("grayscale")
            )
        else:
            noise_dir = find_latest_directory(str(paths["noise"]))
            dataset = MVTecDatasetDenoising(
                root_dir=aug_dir, noise_dir=noise_dir,
                grayscale=self.train_cfg.get("grayscale"),
            )

        dataset_size = len(dataset)
        val_size = int(self.train_cfg.get("validation_split") * dataset_size)
        train_size = dataset_size - val_size

        logging.info(f"Dataset {dataset_size}: train {train_size}, valid {val_size}")

        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_dataloader = DataLoader(
            dataset=train_dataset, batch_size=self.train_cfg.get("batch_size"), shuffle=True
        )
        val_dataloader = DataLoader(
            dataset=val_dataset, batch_size=self.train_cfg.get("batch_size"), shuffle=False
        )

        return train_dataloader, val_dataloader

    def forward_step(self, data):
        """
        Run one forward pass and compute the reconstruction loss.

        Args:
            data: A batch (images for AE/AEE, or (images, noise_images) for DAE/DAEE).

        Returns:
            tuple: (images, recon, loss) for AE/AEE, or (images, noise_images, recon, loss).
        """
        if self.network_type in ["AE", "AEE"]:
            images = data.to(self.device)
            recon = self.model(images)
            loss = 1 - self.criterion(recon, images)
            return images, recon, loss

        images, noise_images = data
        images = images.to(self.device)
        noise_images = noise_images.to(self.device)
        recon = self.model(noise_images)
        loss = 1 - self.criterion(recon, images)
        return images, noise_images, recon, loss

    def train_loop(self, epoch: int, train_losses: list) -> list:
        """
        Run one training epoch.

        Args:
            epoch: Current epoch number.
            train_losses: Accumulator for per-batch training losses.

        Returns:
            list: The updated train_losses.
        """
        self.model.train()
        for batch_idx, data in tqdm(
            enumerate(self.train_dataloader), total=len(self.train_dataloader), desc="Training"
        ):
            results = self.forward_step(data)

            if len(results) == 3:
                images, recon, train_loss = results
            else:
                images, noise_images, recon, train_loss = results

            self.optimizer.zero_grad()
            train_loss.backward()
            self.optimizer.step()
            train_losses.append(train_loss.item())

            if (self.train_cfg.get("vis_during_training") and self.train_cfg.get("vis_interval")
                    and epoch % self.train_cfg.get("vis_interval") == 0 and batch_idx == 0):
                vis_dir = create_save_dirs(
                    directory_path=str(training_testing_paths(self.dataset_type)["training_vis"]),
                    network_type=self.network_type,
                    timestamp=self.timestamp,
                )
                if self.network_type in ["AE", "AEE"]:
                    visualize_images(
                        clean_images=images, outputs=recon,
                        epoch=epoch, batch_idx=batch_idx, dir_path=str(vis_dir),
                    )
                else:
                    visualize_images(
                        clean_images=images, outputs=recon, noise_images=noise_images,
                        epoch=epoch, batch_idx=batch_idx, dir_path=str(vis_dir),
                    )

        return train_losses

    def valid_loop(self, val_losses: list) -> list:
        """
        Run one validation epoch.

        Args:
            val_losses: Accumulator for per-batch validation losses.

        Returns:
            list: The updated val_losses.
        """
        with torch.no_grad():
            for data in tqdm(self.valid_dataloader, total=len(self.valid_dataloader), desc="Validation"):
                results = self.forward_step(data)
                valid_loss = results[2] if len(results) == 3 else results[3]
                val_losses.append(valid_loss.item())

        return val_losses

    def fit(self, progress_callback=None) -> dict:
        """
        Train the model with early stopping and save the best weights.

        Args:
            progress_callback: Optional callable(current, total, phase) invoked
                after every completed epoch so the caller (the Celery task) can
                report training progress. None disables progress reporting.

        Returns:
            dict: Summary with status, best valid loss, epochs run, total epochs,
            whether early stopping was triggered, and the weights path.
        """
        total_epochs = self.train_cfg.get("epochs")

        best_valid_loss = float("inf")
        best_model_path = None
        early_stopping_counter = 0
        early_stopped = False

        train_losses = []
        valid_losses = []
        epoch = 0

        for epoch in tqdm(range(total_epochs), desc="Epochs"):
            train_losses = self.train_loop(epoch, train_losses)
            valid_losses = self.valid_loop(valid_losses)

            if self.train_cfg.get("decrease_learning_rate"):
                self.scheduler.step()

            train_loss = np.average(train_losses)
            valid_loss = np.average(valid_losses)

            self.writer.add_scalars(
                "Loss",
                {
                    "Train": train_loss,
                    "Valid": valid_loss,
                },
                epoch,
            )

            logging.info(f"Train Loss: {train_loss:.5f} valid Loss: {valid_loss:.5f}")

            train_losses.clear()
            valid_losses.clear()

            save_list_to_json(
                filename=self.params_path,
                results_dict={**self.train_cfg, "last_completed_epoch": epoch+1},
            )

            if progress_callback is not None:
                progress_callback(epoch + 1, total_epochs, "epochs")

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                if best_model_path is not None:
                    os.remove(best_model_path)
                best_model_path = os.path.join(str(self.save_path), f"epoch_{epoch}.pt")
                torch.save(self.model.state_dict(), best_model_path)
                logging.info(f"New best weights at epoch {epoch} ({valid_loss:.5f})")
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                logging.warning(f"Early stopping counter: {early_stopping_counter}")
                if early_stopping_counter >= self.train_cfg.get("early_stopping"):
                    logging.info(f"Early stopping at epoch {epoch}")
                    early_stopped = True
                    break

        self.writer.close()
        self.writer.flush()

        return {
            "status": "DONE",
            "network_type": self.network_type,
            "dataset_type": self.dataset_type,
            "best_valid_loss": float(best_valid_loss),
            "epochs_run": epoch + 1,
            "total_epochs": total_epochs,
            "early_stopped": early_stopped,
            "weights_path": best_model_path,
        }