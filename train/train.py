import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from torch.utils.data import DataLoader
from torchsummary import summary

from config.config import ConfigTraining
from config.network_config import network_configs, dataset_images_path_selector
from data_loader import MVTecDataset
from models.network_selector import NetworkFactory


class TrainAutoEncoder:
    def __init__(self):
        self.train_cfg = ConfigTraining().parse()
        network_cfg = network_configs().get(self.train_cfg.network_type)

        dataset = MVTecDataset(root_dir=dataset_images_path_selector().get(self.train_cfg.dataset_type).get("train"))

        self.dataloader = DataLoader(dataset=dataset,
                                     batch_size=self.train_cfg.batch_size,
                                     shuffle=True)
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

    def train(self):
        for epoch in tqdm(range(self.train_cfg.epochs), total=self.train_cfg.epochs, desc="Epochs"):
            self.model.train()  # Set the model to training mode
            total_loss = 0.0

            # Example usage in a training loop
            for batch_idx, images in tqdm(enumerate(self.dataloader), total=len(self.dataloader), desc="Training"):
                images = images.to("cuda")
                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(images)

                # Calculate loss
                loss = self.criterion(outputs, images)

                # Backpropagation
                loss.backward()

                # Update weights
                self.optimizer.step()

                total_loss += loss.item()

                if batch_idx % 10 == 9:  # Print every 10 batches
                    print(f"Epoch [{epoch+1}/{10}], Batch [{batch_idx+1}/{len(self.dataloader)}], "
                          f"Loss: {loss.item():.4f}")

            average_loss = total_loss / len(self.dataloader)
            print(f"Epoch [{epoch+1}/{10}], Average Loss: {average_loss:.4f}")


if __name__ == "__main__":
    ae = TrainAutoEncoder()
    # ae.train()
