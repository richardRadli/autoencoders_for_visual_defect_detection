import colorama
import logging
import numpy as np
import os
import torch
import torch.optim as optim

from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary

from config.network_config import network_configs
from config.dataset_config import dataset_images_path_selector, dataset_data_path_selector
from config.data_paths import JSON_FILES_PATHS
from dataloaders.data_loader_ae import MVTecDataset
from dataloaders.data_loader_dae import MVTecDatasetDenoising
from models.network_selector import NetworkFactory
from utils.utils import create_timestamp, device_selector, setup_logger, get_loss_function, create_save_dirs, \
    visualize_images, load_config_json


class TrainAutoEncoder:
    def __init__(self):
        # Config initialize
        self.timestamp = create_timestamp()
        setup_logger()
        colorama.init()

        self.train_cfg = (
            load_config_json(
                json_schema_filename=JSON_FILES_PATHS.get_data_path("config_schema_training"),
                json_filename=JSON_FILES_PATHS.get_data_path("config_training")
            )
        )

        if self.train_cfg.get("network_type") not in ["AE", "AEE", "DAE", "DAEE"]:
            raise ValueError(f"wrong network type: {self.train_cfg}")

        self.network_type = self.train_cfg.get("network_type")
        self.dataset_type = self.train_cfg.get("dataset_type")

        network_cfg = network_configs(self.train_cfg).get(self.network_type)

        self.device = device_selector(preferred_device=self.train_cfg.get("device"))

        # Setup model
        self.model = (
            NetworkFactory.create_network(
                network_type=self.network_type,
                network_cfg=network_cfg)
        ).to(self.device)

        summary(
            model=self.model,
            input_size=(
                network_cfg.get("input_channel"),
                network_cfg.get("img_size")[0],
                network_cfg.get("img_size")[1]
            )
        )

        self.train_dataloader, self.valid_dataloader = self.create_dataset()

        # Setup loss function, optimizer, LR scheduler and device
        self.criterion = get_loss_function(self.train_cfg.get("loss_function_type"))

        self.optimizer = optim.Adam(params=self.model.parameters(),
                                    lr=self.train_cfg.get("learning_rate"),
                                    weight_decay=self.train_cfg.get("weight_decay"))

        self.scheduler = StepLR(optimizer=self.optimizer,
                                step_size=self.train_cfg.get("step_size"),
                                gamma=self.train_cfg.get("gamma"))

        # Setup directories to save data
        tensorboard_log_dir = (
            create_save_dirs(
                directory_path=dataset_data_path_selector().get(self.dataset_type).get("log_dir"),
                network_type=self.network_type,
                timestamp=self.timestamp
            )
        )

        self.writer = SummaryWriter(log_dir=str(tensorboard_log_dir))

        self.save_path = (
            create_save_dirs(
                directory_path=dataset_data_path_selector().get(self.dataset_type).get("model_weights_dir"),
                network_type=self.network_type,
                timestamp=self.timestamp
            )
        )

    def create_dataset(self):
        """

        :return:
        """

        if self.network_type in ["AE", "AEE"]:
            dataset = (
                MVTecDataset(
                    root_dir=dataset_images_path_selector().get(self.dataset_type).get("aug")
                )
            )
        else:
            dataset = (
                MVTecDatasetDenoising(
                    root_dir=dataset_images_path_selector().get(self.dataset_type).get("aug"),
                    noise_dir=dataset_images_path_selector().get(self.dataset_type).get("noise")
                )
            )

        dataset_size = len(dataset)
        val_size = int(self.train_cfg.get("validation_split") * dataset_size)
        train_size = dataset_size - val_size

        logging.info(f"The size of the dataset is {dataset_size}, "
                     f"from that, the size of the training set is {train_size} "
                     f"and the size of the validation set is {val_size}")

        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_dataloader = DataLoader(dataset=train_dataset, batch_size=self.train_cfg.get("batch_size"), shuffle=True)
        val_dataloader = DataLoader(dataset=val_dataset, batch_size=self.train_cfg.get("batch_size"), shuffle=False)

        return train_dataloader, val_dataloader

    def train_loop(self, epoch: int, train_losses: list) -> list:
        """

        :param epoch:
        :param train_losses:
        :return:
        """

        self.model.train()
        for batch_idx, images in tqdm(enumerate(self.train_dataloader),
                                      total=len(self.train_dataloader),
                                      desc=colorama.Fore.GREEN + "Training"):
            if self.network_type in ["AE", "AEE"]:
                images = images.to(self.device)
                recon = self.model(images)
                train_loss = 1 - self.criterion(recon, images)
            else:
                images, noise_images = images
                images = images.to(self.device)
                noise_images = noise_images.to(self.device)
                recon = self.model(noise_images)
                train_loss = 1 - self.criterion(recon, images)

            self.optimizer.zero_grad()
            train_loss.backward()
            self.optimizer.step()
            train_losses.append(train_loss.item())

            if (self.train_cfg.get("vis_during_training") and (epoch % self.train_cfg.get("vis_interval") == 0) and
                    batch_idx == 0):
                vis_dir = (
                    create_save_dirs(
                        directory_path=dataset_data_path_selector().get(self.dataset_type).get("training_vis"),
                        network_type=self.network_type,
                        timestamp=self.timestamp
                    )
                )

                if self.network_type in ["AE", "AEE"]:
                    visualize_images(
                        clean_images=images,
                        outputs=recon,
                        epoch=epoch,
                        batch_idx=batch_idx,
                        dir_path=str(vis_dir)
                    )
                else:
                    visualize_images(
                        clean_images=images,
                        outputs=recon,
                        noise_images=noise_images,
                        epoch=epoch,
                        batch_idx=batch_idx,
                        dir_path=str(vis_dir)
                    )

        return train_losses

    def valid_loop(self, val_losses):
        with torch.no_grad():
            for batch_idx, data in tqdm(enumerate(self.valid_dataloader),
                                        total=len(self.valid_dataloader),
                                        desc=colorama.Fore.MAGENTA + "Validation"):
                if self.network_type in ["AE", "AEE"]:
                    images = data.to(self.device)
                    outputs = self.model(images)
                    valid_loss = 1 - self.criterion(outputs, images)
                else:
                    images, noise_images = data
                    images = images.to(self.device)
                    noise_images = noise_images.to(self.device)
                    outputs = self.model(noise_images)
                    valid_loss = 1 - self.criterion(outputs, images)

                val_losses.append(valid_loss.item())

        return val_losses

    def fit(self) -> None:
        """
        Train the model.

        :return: None
        """

        best_valid_loss = float('inf')
        best_model_path = None
        early_stopping_counter = 0

        train_losses = []
        valid_losses = []

        for epoch in tqdm(range(self.train_cfg.get("epochs")), desc=colorama.Fore.LIGHTRED_EX + "Epochs"):
            train_losses = self.train_loop(epoch, train_losses)
            valid_losses = self.valid_loop(valid_losses)

            if self.train_cfg.get("decrease_learning_rate"):
                self.scheduler.step()

            train_loss = np.average(train_losses)
            valid_loss = np.average(valid_losses)
            self.writer.add_scalars("Loss", {"Train": train_loss, "Valid": valid_loss}, epoch)

            logging.info(f"Train Loss: {train_loss:.5f} valid Loss: {valid_loss:.5f}")

            train_losses.clear()
            valid_losses.clear()

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                if best_model_path is not None:
                    os.remove(best_model_path)
                best_model_path = os.path.join(str(self.save_path), "epoch_" + str(epoch) + ".pt")
                torch.save(self.model.state_dict(), best_model_path)
                logging.info(f"New weights have been saved at epoch {epoch} with value of {valid_loss:.5f}")
                early_stopping_counter = 0
            else:
                logging.warning(f"No new weights have been saved. Best valid loss was {best_valid_loss:.5f},\n "
                                f"current valid loss is {valid_loss:.5f}")

                early_stopping_counter += 1

                if early_stopping_counter >= self.train_cfg.get("early_stopping"):
                    logging.info(f"Early stopping at epoch {epoch}")
                    break

        self.writer.close()
        self.writer.flush()


if __name__ == "__main__":
    try:
        ae = TrainAutoEncoder()
        ae.fit()
    except KeyboardInterrupt as kie:
        logging.error(kie)
