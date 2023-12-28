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
from torchsummary import summary

from config.config import ConfigTraining
from config.network_config import network_configs, dataset_images_path_selector, dataset_data_path_selector
from data_loader_ae import MVTecDataset
from models.network_selector import NetworkFactory

from utils.utils import create_timestamp, use_gpu_if_available, setup_logger, get_loss_function, create_save_dirs, \
    visualize_images


class TrainAutoEncoder:
    def __init__(self):
        self.train_cfg = ConfigTraining().parse()
        if self.train_cfg.network_type not in ["AE", "AEE"]:
            raise ValueError(f"wrong network type: {self.train_cfg}")

        # Config initialize
        timestamp = create_timestamp()
        setup_logger()
        network_type = self.train_cfg.network_type
        colorama.init()
        network_cfg = network_configs().get(self.train_cfg.network_type)

        # Setup dataset
        dataset = MVTecDataset(root_dir=dataset_images_path_selector().get(self.train_cfg.dataset_type).get("aug"))
        dataset_size = len(dataset)
        val_size = int(self.train_cfg.validation_split * dataset_size)
        train_size = dataset_size - val_size

        logging.info(f"The size of the dataset is {dataset_size}, "
                     f"from that, the size of the training set is {train_size} "
                     f"and the size of the validation set is {val_size}")

        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_dataloader = DataLoader(dataset=train_dataset, batch_size=self.train_cfg.batch_size, shuffle=False)
        val_dataloader = DataLoader(dataset=val_dataset, batch_size=self.train_cfg.batch_size, shuffle=False)

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        # Setup model
        self.model = NetworkFactory.create_network(network_type=self.train_cfg.network_type,
                                                   network_cfg=network_cfg,
                                                   device="cuda")
        summary(
            model=self.model,
            input_size=(
                network_cfg.get("input_channel"), network_cfg.get("img_size")[0], network_cfg.get("img_size")[1]
            )
        )

        # Setup loss function, optimizer, LR scheduler and device
        self.criterion = get_loss_function(self.train_cfg.loss_function_type)

        self.optimizer = optim.Adam(params=self.model.parameters(),
                                    lr=self.train_cfg.learning_rate,
                                    weight_decay=self.train_cfg.weight_decay)

        self.scheduler = StepLR(optimizer=self.optimizer,
                                step_size=self.train_cfg.step_size,
                                gamma=self.train_cfg.gamma)

        self.device = use_gpu_if_available()

        # Setup directories to save data
        tensorboard_log_dir = (
            create_save_dirs(
                directory_path=dataset_data_path_selector().get(self.train_cfg.dataset_type).get("log_dir"),
                network_type=network_type,
                timestamp=timestamp
            )
        )

        self.writer = SummaryWriter(log_dir=str(tensorboard_log_dir))

        self.save_path = (
            create_save_dirs(
                directory_path=dataset_data_path_selector().get(self.train_cfg.dataset_type).get("model_weights_dir"),
                network_type=network_type,
                timestamp=timestamp
            )
        )

    # ------------------------------------------------------------------------------------------------------------------
    # --------------------------------------------------- T R A I N ----------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def train(self):
        """

        :return:
        """

        best_valid_loss = float('inf')
        best_model_path = None

        for epoch in tqdm(
                range(self.train_cfg.epochs), total=self.train_cfg.epochs, desc=colorama.Fore.LIGHTRED_EX + "Epochs"
        ):
            train_losses = []
            valid_losses = []

            self.model.train()
            for batch_idx, images in tqdm(enumerate(self.train_dataloader),
                                          total=len(self.train_dataloader),
                                          desc=colorama.Fore.GREEN + "Training"):
                images = images.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                train_loss = self.criterion(outputs, images)
                train_loss.backward()
                self.optimizer.step()
                train_losses.append(train_loss.item())

                if self.train_cfg.vis_during_training and (epoch % self.train_cfg.vis_interval == 0):
                    visualize_images(clean_images=images,
                                     outputs=outputs,
                                     epoch=epoch,
                                     batch_idx=batch_idx)

            with torch.no_grad():
                for batch_idx, images in tqdm(enumerate(self.val_dataloader),
                                              total=len(self.val_dataloader),
                                              desc=colorama.Fore.MAGENTA + "Validation"):
                    images = images.to(self.device)
                    outputs = self.model(images)
                    valid_loss = self.criterion(outputs, images)
                    valid_losses.append(valid_loss.item())

            if self.train_cfg.decrease_learning_rate:
                self.scheduler.step()

            train_loss_avg = np.average(train_losses)
            valid_loss_avg = np.average(valid_losses)
            self.writer.add_scalars("Loss", {"Train": train_loss_avg, "Valid": valid_loss_avg}, epoch)

            logging.info(f"Train Loss: {train_loss_avg:.5f} valid Loss: {valid_loss_avg:.5f}")

            train_losses.clear()
            valid_losses.clear()

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                if best_model_path is not None:
                    os.remove(best_model_path)
                best_model_path = os.path.join(str(self.save_path), "epoch_" + str(epoch) + ".pt")
                torch.save(self.model.state_dict(), best_model_path)
                logging.info(f"New weights have been saved at epoch {epoch} with value of {valid_loss:.5f}")
            else:
                logging.warning(f"No new weights have been saved. Best valid loss was {best_valid_loss:.5f},\n "
                                f"current valid loss is {valid_loss:.5f}")

        self.writer.close()
        self.writer.flush()


# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------ M A I N -------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        ae = TrainAutoEncoder()
        ae.train()
    except KeyboardInterrupt as kie:
        logging.error(kie)
