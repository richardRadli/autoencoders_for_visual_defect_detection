import colorama
import logging
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

from config.config import ConfigTraining
from config.network_config import network_configs, dataset_images_path_selector
from data_loader import MVTecDataset
from models.network_selector import NetworkFactory
from utils.utils import create_timestamp, use_gpu_if_available, setup_logger


class TrainAutoEncoder:
    def __init__(self):
        self.logger = setup_logger()

        colorama.init()

        self.timestamp = create_timestamp()

        self.train_cfg = ConfigTraining().parse()
        network_cfg = network_configs().get(self.train_cfg.network_type)

        dataset = MVTecDataset(root_dir=dataset_images_path_selector().get(self.train_cfg.dataset_type).get("train"))

        dataset_size = len(dataset)
        val_size = int(self.train_cfg.validation_split * dataset_size)
        train_size = dataset_size - val_size

        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_dataloader = DataLoader(dataset=train_dataset, batch_size=self.train_cfg.batch_size, shuffle=True)
        val_dataloader = DataLoader(dataset=val_dataset, batch_size=self.train_cfg.batch_size, shuffle=False)

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        self.model = NetworkFactory.create_network(network_type=self.train_cfg.network_type,
                                                   network_cfg=network_cfg,
                                                   device="cuda")
        summary(self.model, (3, 128, 128))

        if self.train_cfg.loss_function_type == "mse":
            self.criterion = nn.MSELoss()
        elif self.train_cfg.loss_function_type == "ssim":
            pass
        else:
            raise ValueError(f"Wrong loss function type {self.train_cfg.loss_function_type}")

        self.optimizer = optim.Adam(params=self.model.parameters(),
                                    lr=self.train_cfg.learning_rate,
                                    weight_decay=self.train_cfg.weight_decay)

        self.scheduler = StepLR(optimizer=self.optimizer,
                                step_size=self.train_cfg.step_size,
                                gamma=self.train_cfg.gamma)

        self.device = use_gpu_if_available()

        tensorboard_log_dir = self.create_save_dirs(network_cfg.get('logs_dir'))
        self.writer = SummaryWriter(log_dir=tensorboard_log_dir)

    # ------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------- C R E A T E   S A V E   D I R S -----------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def create_save_dirs(self, network_cfg):
        """

        :param network_cfg:
        :return:
        """

        directory_path = network_cfg.get(self.cfg.type_of_net).get(self.cfg.dataset_type)
        directory_to_create = os.path.join(directory_path, f"{self.timestamp}_{self.cfg.type_of_loss_func}")
        os.makedirs(directory_to_create, exist_ok=True)
        return directory_to_create

    def train(self):
        best_valid_loss = float('inf')
        best_model_path = None

        for epoch in tqdm(range(self.train_cfg.epochs), total=self.train_cfg.epochs, desc="Epochs"):
            train_losses = []
            valid_losses = []

            self.model.train()
            for batch_idx, images in tqdm(enumerate(self.train_dataloader),
                                          total=len(self.train_dataloader),
                                          desc="Training"):
                images = images.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                train_loss = self.criterion(outputs, images)
                train_loss.backward()
                self.optimizer.step()

                train_losses.append(train_loss.item())

            with torch.no_grad():
                for batch_idx, images in tqdm(enumerate(self.val_dataloader),
                                              total=len(self.val_dataloader),
                                              desc="Validation"):
                    images = images.to(self.device)
                    outputs = self.model(images)
                    valid_loss = self.criterion(outputs, images)
                    valid_losses.append(valid_loss.item())

            self.scheduler.step()

            train_loss_avg = np.average(train_losses)
            valid_loss_avg = np.average(valid_losses)
            self.writer.add_scalars("Loss", {"Train": train_loss_avg, "Valid": valid_loss_avg})

            logging.info(f"Train Loss: {train_loss_avg:.5f} valid Loss: {valid_loss_avg:.5f}")

            train_losses.clear()
            valid_losses.clear()

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                if best_model_path is not None:
                    os.remove(best_model_path)
                best_model_path = os.path.join(self.save_path, "epoch_" + str(epoch) + ".pt")
                torch.save(self.model.state_dict(), best_model_path)
                logging.info(f"New weights have been saved at epoch {epoch} with value of {valid_loss:.5f}")
            else:
                logging.warning(f"No new weights have been saved. Best valid loss was {best_valid_loss:.5f},\n "
                                f"current valid loss is {valid_loss:.5f}")

        self.writer.close()
        self.writer.flush()


if __name__ == "__main__":
    try:
        ae = TrainAutoEncoder()
        ae.train()
    except KeyboardInterrupt as kie:
        logging.error(kie)
